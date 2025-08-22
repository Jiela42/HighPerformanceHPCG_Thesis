// This file provides utility functions and GPU kernels for managing halo regions and data exchange
// in a multi-GPU setting using CUDA and MPI. It includes routines for initializing, zeroing,
// injecting, extracting, and verifying halo data.

#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <testing.hpp>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <cassert>
#include <stdbool.h>
#include <thread>

/**
 * @brief Converts a local coordinate index to the corresponding index within a halo data region.
 * @param i Local linear index.
 * @param nx, ny, nz Local grid dimensions.
 * @param dimx, dimy Dimensions of the data region including halos.
 * @return Index into the halo data region.
 */
__inline__ __device__ global_int_t local_i_to_halo_i(
    int i,
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
)
{
    return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

/**
 * @brief Kernel to inject contiguous data into a halo data region.
 * @param x_d Destination halo data array (device).
 * @param data Source contiguous data array (device).
 * @param nx, ny, nz Local grid dimensions.
 * @param dimx, dimy Dimensions of the data region including halos.
 */
__global__ void inject_data_to_halo_kernel(DataType *x_d, DataType *data, int nx, int ny, int nz, int dimx, int dimy)
{
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t n = nx * ny * nz;
    for(local_int_t i = tid; i < n; i += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
        x_d[hi] = data[i];
    }
}

/**
 * @brief Stores and initializes all necessary metadata for the multi-GPU setting.
 *        Stores everything into a Problem struct.
 */
void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem)
{
    problem->npx = npx; //number of processes in x
    problem->npy = npy; //number of processes in y
    problem->npz = npz; //number of processes in z
    problem->nx = nx; //number of grid points of processes subdomain in x
    problem->ny = ny; //number of grid points of processes subdomain in y
    problem->nz = nz; //number of grid points of processes subdomain in z
    local_int_t dimx = nx + 2;
    local_int_t dimy = ny + 2;
    local_int_t dimz = nz + 2;
    problem->size = size;
    problem->rank = rank;
    problem->gnx = npx * nx; //global number of grid points in x
    problem->gny = npy * ny; //global number of grid points in y
    problem->gnz = npz * nz; //global number of grid points in z
    assert(size == npx * npy * npz); //each subdomain of size nx * ny * nz must be assigned to a process
    int px = rank % npx; //x index for this process in process grid
    int py = (rank % (npx * npy)) / npx; //y index for this process in process grid
    int pz = rank / (npx * npy); //z index for this process in process grid
    problem->px = px;
    problem->py = py;
    problem->pz = pz;
    problem->gi0 = (px * nx) + (py * npx * nx * ny) + (pz * npx * npy * nx * ny * nz); //base global index for this rank in the npx by npy by npz point grid layed out in 1D
    problem->gx0 = px * nx; //base global x index for this rank in the npx by npy by npz point grid
    problem->gy0 = py * ny; //base global y index for this rank in the npx by npy by npz point grid
    problem->gz0 = pz * nz; //base global z index for this rank in the npx by npy by npz point grid

    // initialize neighbors, follows the same order as Comm_Tags
    int tmp_neighbors[NUMBER_NEIGHBORS] = {
        /* 0 NORTH       */ (problem->py > 0)                          ? problem->rank - problem->npx               : -1,
        /* 1 EAST        */ (problem->px < problem->npx - 1)             ? problem->rank + 1                          : -1,
        /* 2 SOUTH       */ (problem->py < problem->npy - 1)             ? problem->rank + problem->npx               : -1,
        /* 3 WEST        */ (problem->px > 0)                          ? problem->rank - 1                          : -1,
        /* 4 NE          */ (problem->py > 0 && problem->px < problem->npx - 1) ? problem->rank - problem->npx + 1        : -1,
        /* 5 SE          */ (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) ? problem->rank + problem->npx + 1 : -1,
        /* 6 SW          */ (problem->py < problem->npy - 1 && problem->px > 0) ? problem->rank + problem->npx - 1        : -1,
        /* 7 NW          */ (problem->py > 0 && problem->px > 0)         ? problem->rank - problem->npx - 1             : -1,
        /* 8 FRONT       */ (problem->pz > 0)                          ? problem->rank - problem->npx * problem->npy  : -1,
        /* 9 BACK        */ (problem->pz < problem->npz - 1)             ? problem->rank + problem->npx * problem->npy  : -1,
        /* 10 FRONT_NORTH */ (problem->pz > 0 && problem->py > 0)         ? problem->rank - problem->npx * (problem->npy + 1): -1,
        /* 11 FRONT_EAST  */ (problem->pz > 0 && problem->px < problem->npx - 1) ? problem->rank - problem->npx * problem->npy + 1 : -1,
        /* 12 FRONT_SOUTH */ (problem->pz > 0 && problem->py < problem->npy - 1) ? problem->rank - problem->npx * (problem->npy - 1): -1,
        /* 13 FRONT_WEST  */ (problem->pz > 0 && problem->px > 0)         ? problem->rank - problem->npx * problem->npy - 1 : -1,
        /* 14 BACK_NORTH  */ (problem->pz < problem->npz - 1 && problem->py > 0) ? problem->rank + problem->npx * (problem->npy - 1) : -1,
        /* 15 BACK_EAST   */ (problem->pz < problem->npz - 1 && problem->px < problem->npx - 1) ? problem->rank + problem->npx * problem->npy + 1 : -1,
        /* 16 BACK_SOUTH  */ (problem->pz < problem->npz - 1 && problem->py < problem->npy - 1) ? problem->rank + problem->npx * (problem->npy + 1) : -1,
        /* 17 BACK_WEST   */ (problem->pz < problem->npz - 1 && problem->px > 0) ? problem->rank + problem->npx * problem->npy - 1 : -1,
        /* 18 FRONT_NE    */ (problem->pz > 0 && problem->py > 0 && problem->px < problem->npx - 1) ? problem->rank - problem->npx * (problem->npy + 1) + 1 : -1,
        /* 19 FRONT_SE    */ (problem->pz > 0 && problem->py < problem->npy - 1 && problem->px < problem->npx - 1) ? problem->rank - problem->npx * (problem->npy - 1) + 1 : -1,
        /* 20 FRONT_SW    */ (problem->pz > 0 && problem->py < problem->npy - 1 && problem->px > 0) ? problem->rank - problem->npx * (problem->npy - 1) - 1 : -1,
        /* 21 FRONT_NW    */ (problem->pz > 0 && problem->py > 0 && problem->px > 0) ? problem->rank - problem->npx * (problem->npy + 1) - 1 : -1,
        /* 22 BACK_NE     */ (problem->pz < problem->npz - 1 && problem->py > 0 && problem->px < problem->npx - 1) ? problem->rank + problem->npx * (problem->npy - 1) + 1 : -1,
        /* 23 BACK_SE     */ (problem->pz < problem->npz - 1 && problem->py < problem->npy - 1 && problem->px < problem->npx - 1) ? problem->rank + problem->npx * (problem->npy + 1) + 1 : -1,
        /* 24 BACK_SW     */ (problem->pz < problem->npz - 1 && problem->py < problem->npy - 1 && problem->px > 0) ? problem->rank + problem->npx * (problem->npy + 1) - 1 : -1,
        /* 25 BACK_NW     */ (problem->pz < problem->npz - 1 && problem->py > 0 && problem->px > 0) ? problem->rank + problem->npx * (problem->npy - 1) - 1 : -1
    };
    memcpy(problem->neighbors, tmp_neighbors, sizeof(tmp_neighbors));

    // initialize neighbors_mask, follows the same order as Comm_Tags
    bool tmp_neighbors_mask[NUMBER_NEIGHBORS] = {
        /* NORTH       */ (problem->py > 0),
        /* EAST        */ (problem->px < problem->npx - 1),
        /* SOUTH       */ (problem->py < problem->npy - 1),
        /* WEST        */ (problem->px > 0),
        /* NE          */ (problem->px < problem->npx - 1 && problem->py > 0),
        /* SE          */ (problem->px < problem->npx - 1 && problem->py < problem->npy - 1),
        /* SW          */ (problem->px > 0 && problem->py < problem->npy - 1),
        /* NW          */ (problem->px > 0 && problem->py > 0),
        /* FRONT       */ (problem->pz > 0),
        /* BACK        */ (problem->pz < problem->npz - 1),
        /* FRONT_NORTH */ (problem->py > 0 && problem->pz > 0),
        /* FRONT_EAST  */ (problem->px < problem->npx - 1 && problem->pz > 0),
        /* FRONT_SOUTH */ (problem->py < problem->npy - 1 && problem->pz > 0),
        /* FRONT_WEST  */ (problem->px > 0 && problem->pz > 0),
        /* BACK_NORTH  */ (problem->py > 0 && problem->pz < problem->npz - 1),
        /* BACK_EAST   */ (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1),
        /* BACK_SOUTH  */ (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1),
        /* BACK_WEST   */ (problem->px > 0 && problem->pz < problem->npz - 1),
        /* FRONT_NE    */ (problem->pz > 0 && problem->py > 0 && problem->px < problem->npx - 1),
        /* FRONT_SE    */ (problem->pz > 0 && problem->py < problem->npy - 1 && problem->px < problem->npx - 1),
        /* FRONT_SW    */ (problem->pz > 0 && problem->py < problem->npy - 1 && problem->px > 0),
        /* FRONT_NW    */ (problem->pz > 0 && problem->py > 0 && problem->px > 0),
        /* BACK_NE     */ (problem->pz < problem->npz - 1 && problem->py > 0 && problem->px < problem->npx - 1),
        /* BACK_SE     */ (problem->pz < problem->npz - 1 && problem->py < problem->npy - 1 && problem->px < problem->npx - 1),
        /* BACK_SW     */ (problem->pz < problem->npz - 1 && problem->py < problem->npy - 1 && problem->px > 0),
        /* BACK_NW     */ (problem->pz < problem->npz - 1 && problem->py > 0 && problem->px > 0)
    };
    memcpy(problem->neighbors_mask, tmp_neighbors_mask, sizeof(tmp_neighbors_mask));

    // initialize count_exchange, follows the same order as Comm_Tags
    local_int_t tmp_count_exchange[NUMBER_NEIGHBORS] = {
        /* NORTH       */ nx * nz,
        /* EAST        */ ny * nz,
        /* SOUTH       */ nx * nz,
        /* WEST        */ ny * nz,
        /* NE          */ nz,
        /* SE          */ nz,
        /* SW          */ nz,
        /* NW          */ nz,
        /* FRONT       */ nx * ny,
        /* BACK        */ nx * ny,
        /* FRONT_NORTH */ nx,
        /* FRONT_EAST  */ ny,
        /* FRONT_SOUTH */ nx,
        /* FRONT_WEST  */ ny,
        /* BACK_NORTH  */ nx,
        /* BACK_EAST   */ ny,
        /* BACK_SOUTH  */ nx,
        /* BACK_WEST   */ ny,
        /* FRONT_NE    */ 1,
        /* FRONT_SE    */ 1,
        /* FRONT_SW    */ 1,
        /* FRONT_NW    */ 1,
        /* BACK_NE     */ 1,
        /* BACK_SE     */ 1,
        /* BACK_SW     */ 1,
        /* BACK_NW     */ 1
    };
    memcpy(problem->count_exchange, tmp_count_exchange, sizeof(tmp_count_exchange));

    // initialize the Ghost Cells, which store information of the correct extraction and injection from/to GPU
    GhostCell tmp_extraction_ghost_cells[NUMBER_NEIGHBORS] = {
        // NORTH: extract_horizontal_plane from (1,1,1) with patch (nx, 1, nz)
        { 1, 1, 1,    dimx, dimy, dimz,    nx, 1, nz },
        
        // EAST: extract_vertical_plane from (dimx-2,1,1) with patch (1, ny, nz)
        { dimx - 2, 1, 1,    dimx, dimy, dimz,    1, ny, nz },
        
        // SOUTH: extract_horizontal_plane from (1,dimy-2,1) with patch (nx, 1, nz)
        { 1, dimy - 2, 1,   dimx, dimy, dimz,    nx, 1, nz },
        
        // WEST: extract_vertical_plane from (1,1,1) with patch (1, ny, nz)
        { 1, 1, 1,    dimx, dimy, dimz,    1, ny, nz },
        
        // NE: extract_edge_Z from (dimx-2,1,1) with patch (1, 1, nz)
        { dimx - 2, 1, 1,    dimx, dimy, dimz,    1, 1, nz },
        
        // SE: extract_edge_Z from (dimx-2, dimy-2,1) with patch (1, 1, nz)
        { dimx - 2, dimy - 2, 1,   dimx, dimy, dimz,    1, 1, nz },
        
        // SW: extract_edge_Z from (1, dimy-2,1) with patch (1, 1, nz)
        { 1, dimy - 2, 1,   dimx, dimy, dimz,    1, 1, nz },
        
        // NW: extract_edge_Z from (1,1,1) with patch (1, 1, nz)
        { 1, 1, 1,    dimx, dimy, dimz,    1, 1, nz },
        
        // FRONT: extract_frontal_plane from (1,1,1) with patch (nx, ny, 1)
        { 1, 1, 1,    dimx, dimy, dimz,    nx, ny, 1 },
        
        // BACK: extract_frontal_plane from (1,1, dimz-2) with patch (nx, ny, 1)
        { 1, 1, dimz - 2,   dimx, dimy, dimz,    nx, ny, 1 },
        
        // FRONT_NORTH: extract_edge_X from (1,1,1) with patch (nx, 1, 1)
        { 1, 1, 1,    dimx, dimy, dimz,    nx, 1, 1 },
        
        // FRONT_EAST: extract_edge_Y from (dimx-2,1,1) with patch (1, ny, 1)
        { dimx - 2, 1, 1,    dimx, dimy, dimz,    1, ny, 1 },
        
        // FRONT_SOUTH: extract_edge_X from (1, dimy-2,1) with patch (nx, 1, 1)
        { 1, dimy - 2, 1,    dimx, dimy, dimz,    nx, 1, 1 },
        
        // FRONT_WEST: extract_edge_Y from (1,1,1) with patch (1, ny, 1)
        { 1, 1, 1,    dimx, dimy, dimz,    1, ny, 1 },
        
        // BACK_NORTH: extract_edge_X from (1,1, dimz-2) with patch (nx, 1, 1)
        { 1, 1, dimz - 2,   dimx, dimy, dimz,    nx, 1, 1 },
        
        // BACK_EAST: extract_edge_Y from (dimx-2,1, dimz-2) with patch (1, ny, 1)
        { dimx - 2, 1, dimz - 2,   dimx, dimy, dimz,    1, ny, 1 },
        
        // BACK_SOUTH: extract_edge_X from (1, dimy-2, dimz-2) with patch (nx, 1, 1)
        { 1, dimy - 2, dimz - 2,   dimx, dimy, dimz,    nx, 1, 1 },
        
        // BACK_WEST: extract_edge_Y from (1,1, dimz-2) with patch (1, ny, 1)
        { 1, 1, dimz - 2,   dimx, dimy, dimz,    1, ny, 1 },
        
        // FRONT_NE: corner extraction from (dimx-2,1,1) with patch (1, 1, 1)
        { dimx - 2, 1, 1,    dimx, dimy, dimz,    1, 1, 1 },
        
        // FRONT_SE: corner extraction from (dimx-2, dimy-2,1) with patch (1, 1, 1)
        { dimx - 2, dimy - 2, 1,   dimx, dimy, dimz,    1, 1, 1 },
        
        // FRONT_SW: corner extraction from (1, dimy-2,1) with patch (1, 1, 1)
        { 1, dimy - 2, 1,   dimx, dimy, dimz,    1, 1, 1 },
        
        // FRONT_NW: corner extraction from (1,1,1) with patch (1, 1, 1)
        { 1, 1, 1,    dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_NE: corner extraction from (dimx-2,1, dimz-2) with patch (1, 1, 1)
        { dimx - 2, 1, dimz - 2,   dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_SE: corner extraction from (dimx-2, dimy-2, dimz-2) with patch (1, 1, 1)
        { dimx - 2, dimy - 2, dimz - 2,   dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_SW: corner extraction from (1, dimy-2, dimz-2) with patch (1, 1, 1)
        { 1, dimy - 2, dimz - 2,   dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_NW: corner extraction from (1,1, dimz-2) with patch (1, 1, 1)
        { 1, 1, dimz - 2,   dimx, dimy, dimz,    1, 1, 1 }
    };
    memcpy(problem->extraction_ghost_cells, tmp_extraction_ghost_cells, sizeof(tmp_extraction_ghost_cells));

    GhostCell tmp_injection_ghost_cells[NUMBER_NEIGHBORS] = {
        // NORTH: inject_horizontal_plane_to_GPU(x_d, halo->north_recv_buff_h, 1, 0, 1, nx, nz, dimx, dimy, dimz);
        { 1, 0, 1,    dimx, dimy, dimz,    nx, 1, nz },
        
        // EAST: inject_vertical_plane_to_GPU(x_d, halo->east_recv_buff_h, dimx - 1, 1, 1, ny, nz, dimx, dimy, dimz);
        { dimx - 1, 1, 1,    dimx, dimy, dimz,    1, ny, nz },
        
        // SOUTH: inject_horizontal_plane_to_GPU(x_d, halo->south_recv_buff_h, 1, dimy - 1, 1, nx, nz, dimx, dimy, dimz);
        { 1, dimy - 1, 1,    dimx, dimy, dimz,    nx, 1, nz },
        
        // WEST: inject_vertical_plane_to_GPU(x_d, halo->west_recv_buff_h, 0, 1, 1, ny, nz, dimx, dimy, dimz);
        { 0, 1, 1,    dimx, dimy, dimz,    1, ny, nz },
        
        // NE: inject_edge_Z_to_GPU(x_d, halo->ne_recv_buff_h, dimx - 1, 0, 1, nz, dimx, dimy, dimz);
        { dimx - 1, 0, 1,    dimx, dimy, dimz,    1, 1, nz },
        
        // SE: inject_edge_Z_to_GPU(x_d, halo->se_recv_buff_h, dimx - 1, dimy - 1, 1, nz, dimx, dimy, dimz);
        { dimx - 1, dimy - 1, 1,    dimx, dimy, dimz,    1, 1, nz },
        
        // SW: inject_edge_Z_to_GPU(x_d, halo->sw_recv_buff_h, 0, dimy - 1, 1, nz, dimx, dimy, dimz);
        { 0, dimy - 1, 1,    dimx, dimy, dimz,    1, 1, nz },
        
        // NW: inject_edge_Z_to_GPU(x_d, halo->nw_recv_buff_h, 0, 0, 1, nz, dimx, dimy, dimz);
        { 0, 0, 1,    dimx, dimy, dimz,    1, 1, nz },
        
        // FRONT: inject_frontal_plane_to_GPU(x_d, halo->front_recv_buff_h, 1, 1, 0, nx, ny, dimx, dimy, dimz);
        { 1, 1, 0,    dimx, dimy, dimz,    nx, ny, 1 },
        
        // BACK: inject_frontal_plane_to_GPU(x_d, halo->back_recv_buff_h, 1, 1, dimz - 1, nx, ny, dimx, dimy, dimz);
        { 1, 1, dimz - 1,    dimx, dimy, dimz,    nx, ny, 1 },
        
        // FRONT_NORTH: inject_edge_X_to_GPU(x_d, halo->front_north_recv_buff_h, 1, 0, 0, nx, dimx, dimy, dimz);
        { 1, 0, 0,    dimx, dimy, dimz,    nx, 1, 1 },
        
        // FRONT_EAST: inject_edge_Y_to_GPU(x_d, halo->front_east_recv_buff_h, dimx - 1, 1, 0, ny, dimx, dimy, dimz);
        { dimx - 1, 1, 0,    dimx, dimy, dimz,    1, ny, 1 },
        
        // FRONT_SOUTH: inject_edge_X_to_GPU(x_d, halo->front_south_recv_buff_h, 1, dimy - 1, 0, nx, dimx, dimy, dimz);
        { 1, dimy - 1, 0,    dimx, dimy, dimz,    nx, 1, 1 },
        
        // FRONT_WEST: inject_edge_Y_to_GPU(x_d, halo->front_west_recv_buff_h, 0, 1, 0, ny, dimx, dimy, dimz);
        { 0, 1, 0,    dimx, dimy, dimz,    1, ny, 1 },
        
        // BACK_NORTH: inject_edge_X_to_GPU(x_d, halo->back_north_recv_buff_h, 1, 0, dimz - 1, nx, dimx, dimy, dimz);
        { 1, 0, dimz - 1,    dimx, dimy, dimz,    nx, 1, 1 },
        
        // BACK_EAST: inject_edge_Y_to_GPU(x_d, halo->back_east_recv_buff_h, dimx - 1, 1, dimz - 1, ny, dimx, dimy, dimz);
        { dimx - 1, 1, dimz - 1,    dimx, dimy, dimz,    1, ny, 1 },
        
        // BACK_SOUTH: inject_edge_X_to_GPU(x_d, halo->back_south_recv_buff_h, 1, dimy - 1, dimz - 1, nx, dimx, dimy, dimz);
        { 1, dimy - 1, dimz - 1,    dimx, dimy, dimz,    nx, 1, 1 },
        
        // BACK_WEST: inject_edge_Y_to_GPU(x_d, halo->back_west_recv_buff_h, 0, 1, dimz - 1, ny, dimx, dimy, dimz);
        { 0, 1, dimz - 1,    dimx, dimy, dimz,    1, ny, 1 },
        
        // FRONT_NE (corner injection): corresponds to cudaMemcpy(x_d + dimx - 1, ...),
        // which gives coordinate (dimx - 1, 0, 0)
        { dimx - 1, 0, 0,    dimx, dimy, dimz,    1, 1, 1 },
        
        // FRONT_SE (corner injection): corresponds to cudaMemcpy(x_d + dimx * dimy - 1, ...),
        // i.e. (dimx - 1, dimy - 1, 0)
        { dimx - 1, dimy - 1, 0,    dimx, dimy, dimz,    1, 1, 1 },
        
        // FRONT_SW (corner injection): corresponds to cudaMemcpy(x_d + (dimy - 1) * dimx, ...),
        // i.e. (0, dimy - 1, 0)
        { 0, dimy - 1, 0,    dimx, dimy, dimz,    1, 1, 1 },
        
        // FRONT_NW (corner injection): corresponds to cudaMemcpy(x_d, ...),
        // i.e. (0, 0, 0)
        { 0, 0, 0,    dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_NE (corner injection): corresponds to cudaMemcpy(x_d + dimx - 1 + dimx * dimy * (dimz - 1), ...),
        // i.e. (dimx - 1, 0, dimz - 1)
        { dimx - 1, 0, dimz - 1,    dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_SE (corner injection): corresponds to cudaMemcpy(x_d + dimx - 1 + (dimy - 1) * dimx + dimx * dimy * (dimz - 1), ...),
        // i.e. (dimx - 1, dimy - 1, dimz - 1)
        { dimx - 1, dimy - 1, dimz - 1,    dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_SW (corner injection): corresponds to cudaMemcpy(x_d + dimx * (dimy - 1) + dimx * dimy * (dimz - 1), ...),
        // i.e. (0, dimy - 1, dimz - 1)
        { 0, dimy - 1, dimz - 1,    dimx, dimy, dimz,    1, 1, 1 },
        
        // BACK_NW (corner injection): corresponds to cudaMemcpy(x_d + dimx * dimy * (dimz - 1), ...),
        // i.e. (0, 0, dimz - 1)
        { 0, 0, dimz - 1,    dimx, dimy, dimz,    1, 1, 1 }
    };
    memcpy(problem->injection_ghost_cells, tmp_injection_ghost_cells, sizeof(tmp_injection_ghost_cells));

    void (*tmp_extraction_functions[NUMBER_NEIGHBORS])(Halo *halo, int i_buff, GhostCell *gh, bool host_buff) = {
        extract_horizontal_plane_from_GPU,
        extract_vertical_plane_from_GPU,
        extract_horizontal_plane_from_GPU,
        extract_vertical_plane_from_GPU,
        extract_edge_Z_from_GPU,
        extract_edge_Z_from_GPU,
        extract_edge_Z_from_GPU,
        extract_edge_Z_from_GPU,
        extract_frontal_plane_from_GPU,
        extract_frontal_plane_from_GPU,
        extract_edge_X_from_GPU,
        extract_edge_Y_from_GPU,
        extract_edge_X_from_GPU,
        extract_edge_Y_from_GPU,
        extract_edge_X_from_GPU,
        extract_edge_Y_from_GPU,
        extract_edge_X_from_GPU,
        extract_edge_Y_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU,
        extract_corner_from_GPU
    };
    memcpy(problem->extraction_functions, tmp_extraction_functions, sizeof(tmp_extraction_functions));

    void (*tmp_injection_functions[NUMBER_NEIGHBORS])(Halo *halo, int i_buff, GhostCell *gh, bool host_buff) = {
        inject_horizontal_plane_to_GPU,
        inject_vertical_plane_to_GPU,
        inject_horizontal_plane_to_GPU,
        inject_vertical_plane_to_GPU,
        inject_edge_Z_to_GPU,
        inject_edge_Z_to_GPU,
        inject_edge_Z_to_GPU,
        inject_edge_Z_to_GPU,
        inject_frontal_plane_to_GPU,
        inject_frontal_plane_to_GPU,
        inject_edge_X_to_GPU,
        inject_edge_Y_to_GPU,
        inject_edge_X_to_GPU,
        inject_edge_Y_to_GPU,
        inject_edge_X_to_GPU,
        inject_edge_Y_to_GPU,
        inject_edge_X_to_GPU,
        inject_edge_Y_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU,
        inject_corner_to_GPU
    };
    memcpy(problem->injection_functions, tmp_injection_functions, sizeof(tmp_injection_functions));


}

/**
 * @brief Initializes the variables of a halo struct and allocates all GPU memory necessary for halo exchange and computation.
 *        GPU memory is initialized with zeros.
 */
void InitHaloMemGPU(Halo *halo, Problem *problem)
{
    local_int_t nx = problem->nx;
    local_int_t ny = problem->ny;
    local_int_t nz = problem->nz;
    local_int_t dimx = nx + 2;
    local_int_t dimy = ny + 2;
    local_int_t dimz = nz + 2;
    halo->nx = nx;
    halo->ny = ny;
    halo->nz = nz;
    halo->dimx = dimx;
    halo->dimy = dimy;
    halo->dimz = dimz;
    
    DataType *x_d;
    CHECK_CUDA(cudaMalloc(&x_d, dimx * dimy * dimz * sizeof(DataType)));
    CHECK_CUDA(cudaMemset(x_d, 0, dimx * dimy * dimz * sizeof(DataType)));
    halo->x_d = x_d;
    DataType *interior = x_d + dimx * dimy + dimx + 1;
    halo->interior = interior;

    //allocate communcation buffers on device
    local_int_t count_exchange = 0;
    for(int i = 0; i < NUMBER_NEIGHBORS; i++){
        count_exchange += problem->count_exchange[i];
    }
    DataType *send_buff_d;
    DataType *recv_buff_d;
    CHECK_CUDA(cudaMalloc(&(send_buff_d), count_exchange * sizeof(DataType)));
    CHECK_CUDA(cudaMemset(send_buff_d, 0, count_exchange * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&(recv_buff_d),count_exchange * sizeof(DataType)));
    CHECK_CUDA(cudaMemset(recv_buff_d, 0, count_exchange * sizeof(DataType)));
    for(int i = 0; i < NUMBER_NEIGHBORS; i++){
        halo->send_buff_d[i] = send_buff_d;
        send_buff_d += problem->count_exchange[i];
        halo->recv_buff_d[i] = recv_buff_d;
        recv_buff_d += problem->count_exchange[i];
    }
    // Ensure all device memory setting is complete
    CHECK_CUDA(cudaDeviceSynchronize());
    
}

/**
 * @brief Initializes all halo struct variables and the necessary data regions on the CPU for communication and computation.
 */
void InitHaloMemCPU(Halo *halo, Problem *problem)
{
    local_int_t nx = problem->nx;
    local_int_t ny = problem->ny;
    local_int_t nz = problem->nz;
    local_int_t dimx = nx + 2;
    local_int_t dimy = ny + 2;
    local_int_t dimz = nz + 2;
    halo->nx = nx;
    halo->ny = ny;
    halo->nz = nz;
    halo->dimx = dimx;
    halo->dimy = dimy;
    halo->dimz = dimz;

    // allocate memory for send and receive buffers on CPU
    local_int_t count_exchange = 0;
    for(int i = 0; i < NUMBER_NEIGHBORS; i++){
        count_exchange += problem->count_exchange[i];
    }
    DataType *send_buff_h = (DataType *) calloc(count_exchange, sizeof(DataType));
    DataType *recv_buff_h = (DataType *) calloc(count_exchange, sizeof(DataType));
    for(int i = 0; i < NUMBER_NEIGHBORS; i++){
        halo->send_buff_h[i] = send_buff_h;
        send_buff_h += problem->count_exchange[i];
        halo->recv_buff_h[i] = recv_buff_h;
        recv_buff_h += problem->count_exchange[i];
    }

    // create cuda streams
    int least_priority, greatest_priority;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    halo->least_priority = least_priority;
    halo->greatest_priority = greatest_priority;
    for(int i = 0; i < 27; i++){
        halo->streams[i] = (cudaStream_t*) malloc(sizeof(cudaStream_t));
        CHECK_CUDA(cudaStreamCreateWithPriority(halo->streams[i], cudaStreamNonBlocking, greatest_priority));
    }
    halo->streams[27] = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    CHECK_CUDA(cudaStreamCreateWithPriority(halo->streams[27], cudaStreamNonBlocking, least_priority)); //give less priority to stream for inner computation

}

/**
 * @brief Initializes the memory for halo on both CPU and GPU and initializes all data memory with zeros.
 */
void InitHalo(Halo *halo, Problem *problem)
{
    InitHaloMemGPU(halo, problem);
    InitHaloMemCPU(halo, problem);
}

/**
 * @brief Sets the entire halo data region to zero on the GPU.
 */
void SetHaloZeroGPU(Halo *halo)
{
    CHECK_CUDA(cudaMemset(halo->x_d, 0, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType)));
}

/**
 * @brief Injects a data array of length nx*ny*nz into the data field of a halo struct.
 */
void InjectDataToHalo(Halo *halo, DataType *data)
{
    int n = halo->nx * halo->ny * halo->nz;
    int const nthread = 1024; // number of threads per block
    int const nblock = (n + nthread - 1) / nthread; // total blocks needed
    inject_data_to_halo_kernel<<<nblock, nthread>>>(halo->x_d, data, halo->nx, halo->ny, halo->nz, halo->dimx, halo->dimy);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief Initialize the data region of a halo struct with global index.
 */
void SetHaloGlobalIndexGPU(Halo *halo, Problem *problem)
{
    DataType *x_h = (DataType*) malloc(halo->dimx * halo->dimy * halo->dimz * sizeof(DataType));
    for(int i=0; i<halo->dimx * halo->dimy * halo->dimz; i++){
        x_h[i] = 0;
    }
    DataType *write_addr = x_h+ halo->dimx * halo->dimy + halo->dimx + 1;
    int gi = problem->gi0;
    for(int i = 0; i<halo->nz; i++){
        for(int j = 0; j<halo->ny; j++){
            for(int l = 0; l<halo->nx; l++){
                *write_addr = gi;
                gi++;
                write_addr++;
            }
            write_addr += 2;
            gi = gi - halo->nx + problem->gnx;
        }
        write_addr += 2 * halo->dimx;
        gi = problem->gi0 + (i + 1) * problem->gnx * problem->gny;
    }
    CHECK_CUDA(cudaMemcpy(halo->x_d, x_h, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType), cudaMemcpyHostToDevice));
}

/**
 * @brief Initialize the data region of a halo with 1.0/(global index + 1.0).
 */
void SetHaloQuotientGlobalIndexGPU(Halo *halo, Problem *problem)
{
    DataType *x_h = (DataType*) malloc(halo->dimx * halo->dimy * halo->dimz * sizeof(DataType));
    for(int i=0; i<halo->dimx * halo->dimy * halo->dimz; i++){
        x_h[i] = 0;
    }
    DataType *write_addr = x_h+ halo->dimx * halo->dimy + halo->dimx + 1;
    int gi = problem->gi0;
    for(int i = 0; i<halo->nz; i++){
        for(int j = 0; j<halo->ny; j++){
            for(int l = 0; l<halo->nx; l++){
                *write_addr = 1.0/(gi+1.0);
                gi++;
                write_addr++;
            }
            write_addr += 2;
            gi = gi - halo->nx + problem->gnx;
        }
        write_addr += 2 * halo->dimx;
        gi = problem->gi0 + (i + 1) * problem->gnx * problem->gny;
    }
    CHECK_CUDA(cudaMemcpy(halo->x_d, x_h, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType), cudaMemcpyHostToDevice));
}

/**
 * @brief Initialize the halo with random numbers between min and max.
 *        Seed = input seed + rank for each process.
 */
void SetHaloRandomGPU(Halo *halo, Problem *problem, int min, int max, int seed)
{
    DataType *x_h = (DataType*) malloc(halo->dimx * halo->dimy * halo->dimz * sizeof(DataType));
    for(int i=0; i<halo->dimx * halo->dimy * halo->dimz; i++){
        x_h[i] = 0;
    }
    srand(seed + problem->rank);
    DataType *write_addr = x_h+ halo->dimx * halo->dimy + halo->dimx + 1;
    int gi = problem->gi0;
    for(int i = 0; i<halo->nz; i++){
        for(int j = 0; j<halo->ny; j++){
            for(int l = 0; l<halo->nx; l++){
                int rand_num = rand();
                if(min == 0 && max == 1.0) {
                    *write_addr = (DataType) rand_num / RAND_MAX;
                }else{
                    *write_addr = min + rand_num % (max - min);
                }
                gi++;
                write_addr++;
            }
            write_addr += 2;
            gi = gi - halo->nx + problem->gnx;
        }
        write_addr += 2 * halo->dimx;
        gi = problem->gi0 + (i + 1) * problem->gnx * problem->gny;
    }
    CHECK_CUDA(cudaMemcpy(halo->x_d, x_h, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType), cudaMemcpyHostToDevice));
}

/**
 * @brief Frees all initialized GPU memory for the halo struct.
 */
void FreeHaloGPU(Halo *halo)
{
    CHECK_CUDA(cudaFree(halo->x_d));
    CHECK_CUDA(cudaFree(halo->send_buff_d[0]));
    CHECK_CUDA(cudaFree(halo->recv_buff_d[0]));
}

/**
 * @brief Frees all initialized CPU memory for the halo struct.
 */
void FreeHaloCPU(Halo *halo)
{
    free(halo->send_buff_h[0]);
    free(halo->recv_buff_h[0]);
}

/**
 * @brief Frees both CPU and GPU memory for the halo struct.
 */
void FreeHalo(Halo *halo)
{
    FreeHaloGPU(halo);
    FreeHaloCPU(halo);
}

/**
 * @brief In case there are more than one GPUs visible to the process, picks the one (rank % visibleDevices).
 */
void InitGPU(Problem *problem)
{
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount > 0);
    CHECK_CUDA(cudaSetDevice(problem->rank % deviceCount));
    // printf("Rank=%d:\t\t Set my device to device=%d, available=%d.\n", problem->rank, problem->rank % deviceCount, deviceCount);
}

// ---------------------------------------------------------------------------
// GPU Kernels for extracting/injecting halo planes
// ---------------------------------------------------------------------------

/**
 * @brief Extracts an XZ plane from x_d into slice_d.
 */
__global__ void extract_xz_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Z, int slice_X, int slice_Z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int z_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (z_loc < length_Z) slice_d[tid] = x_d[z_loc*slice_Z + x_loc*slice_X];
}

/**
 * @brief Injects an XZ plane from slice_d into x_d.
 */
__global__ void inject_xz_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Z, int slice_X, int slice_Z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int z_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (z_loc < length_Z) x_d[z_loc*slice_Z + x_loc*slice_X] = slice_d[tid];
}

/**
 * @brief Extracts a YZ plane from x_d into slice_d.
 */
__global__ void extract_yz_plane_kernel(DataType *x_d, DataType *slice_d, int length_Y, int length_Z, int slice_Y, int slice_Z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int z_loc = tid / length_Y;
    int y_loc = tid % length_Y;
    if (z_loc < length_Z) slice_d[tid] = x_d[z_loc*slice_Z + y_loc*slice_Y];
}

/**
 * @brief Injects a YZ plane from slice_d into x_d.
 */
__global__ void inject_yz_plane_kernel(DataType *x_d, DataType *slice_d, int length_Y, int length_Z, int slice_Y, int slice_Z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int z_loc = tid / length_Y;
    int y_loc = tid % length_Y;
    if (z_loc < length_Z) x_d[z_loc*slice_Z + y_loc*slice_Y] = slice_d[tid];
}

/**
 * @brief Extracts an XY plane from x_d into slice_d.
 */
__global__ void extract_xy_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Y, int slice_X, int slice_Y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (y_loc < length_Y) slice_d[tid] = x_d[y_loc*slice_Y + x_loc*slice_X];
}

/**
 * @brief Injects an XY plane from slice_d into x_d.
 */
__global__ void inject_xy_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Y, int slice_X, int slice_Y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (y_loc < length_Y) x_d[y_loc*slice_Y + x_loc*slice_X] = slice_d[tid];
}

/**
 * @brief Extracts a horizontal (XZ) plane from the GPU halo region into a buffer.
 */
void extract_horizontal_plane_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->send_buff_d[i_buff];
    // Kernel launch config for extracting XZ plane:
    int const nthread = 256; // number of threads per block
    int const nblock = (gh->length_X * gh->length_Z + nthread - 1) / nthread; // total blocks needed
    extract_xz_plane_kernel<<<nblock, nthread, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_X, gh->length_Z, 1, gh->dimy * gh->dimx);

    // copy from device to host if needed
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], buff, gh->length_X * gh->length_Z * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Injects a horizontal (XZ) plane from a buffer into the GPU halo region.
 */
void inject_horizontal_plane_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x +  gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType* x_d = halo->x_d + k;
    DataType *buff = halo->recv_buff_d[i_buff];
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(buff, halo->recv_buff_h[i_buff], gh->length_X * gh->length_Z * sizeof(DataType), cudaMemcpyHostToDevice));
    }
    // Kernel launch config for injecting XZ plane:
    int const nthread = 256; // number of threads per block
    int const nblock = (gh->length_X * gh->length_Z + nthread - 1) / nthread; // total blocks needed
    inject_xz_plane_kernel<<<nblock, nthread, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_X, gh->length_Z, 1, gh->dimy * gh->dimx);
}

/**
 * @brief Extracts a vertical (YZ) plane from the GPU halo region into a buffer.
 */
void extract_vertical_plane_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->send_buff_d[i_buff];
    // Kernel launch config for extracting YZ plane:
    int const nthread = 256; // number of threads per block
    int const nblock = (gh->length_Y * gh->length_Z + nthread - 1) / nthread; // total blocks needed
    extract_yz_plane_kernel<<<nblock, nthread, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_Y, gh->length_Z, gh->dimx, gh->dimy * gh->dimx);
    // copy from device to host if needed
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], buff, gh->length_Y * gh->length_Z * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Injects a vertical (YZ) plane from a buffer into the GPU halo region.
 */
void inject_vertical_plane_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x +  gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->recv_buff_d[i_buff];
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(buff, halo->recv_buff_h[i_buff], gh->length_Y * gh->length_Z * sizeof(DataType), cudaMemcpyHostToDevice));
    }
    // Kernel launch config for injecting YZ plane:
    int const nthread = 256; // number of threads per block
    int const nblock = (gh->length_Y * gh->length_Z + nthread - 1) / nthread; // total blocks needed
    inject_yz_plane_kernel<<<nblock, nthread, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_Y, gh->length_Z, gh->dimx, gh->dimy * gh->dimx);
}

/**
 * @brief Extracts a frontal (XY) plane from the GPU halo region into a buffer.
 */
void extract_frontal_plane_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x +  gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->send_buff_d[i_buff];
    // Kernel launch config for extracting XY plane:
    int const nthread = 256; // number of threads per block
    int const nblock = (gh->length_X * gh->length_Y + nthread - 1) / nthread; // total blocks needed
    extract_xy_plane_kernel<<<nblock, nthread, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_X, gh->length_Y, 1, gh->dimx);
    // copy from device to host
    if(host_buff) {
        // Ensure the data is copied back to host memory
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], buff, gh->length_X * gh->length_Y * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Injects a frontal (XY) plane from a buffer into the GPU halo region.
 */
void inject_frontal_plane_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x +  gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->recv_buff_d[i_buff];
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(buff, halo->recv_buff_h[i_buff], gh->length_Y * gh->length_X * sizeof(DataType), cudaMemcpyHostToDevice));
    }
    // Kernel launch config for injecting XY plane:
    int const nthread = 256; // number of threads per block
    int const nblock = (gh->length_X * gh->length_Y + nthread - 1) / nthread; // total blocks needed
    inject_xy_plane_kernel<<<nblock, nthread, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_X, gh->length_Y, 1, gh->dimx);
}

/**
 * @brief Kernel to extract a 1D edge from x_d into slice_d.
 */
__global__ void extract_edge_kernel(DataType *x_d, DataType *slice_d, int length_X, int slice_X)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length_X) slice_d[tid] = x_d[tid * slice_X];
}

/**
 * @brief Kernel to inject a 1D edge from slice_d into x_d.
 */
__global__ void inject_edge_kernel(DataType *x_d, DataType *slice_d, int length_X, int slice_X)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length_X) x_d[tid * slice_X] = slice_d[tid];
}

/**
 * @brief Extracts an edge along X from the GPU halo region into a buffer.
 */
void extract_edge_X_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->send_buff_d[i_buff];
    int const nthreads = 128; // number of threads per block
    int const nblocks = (gh->length_X + nthreads - 1) / nthreads; // total blocks needed
    extract_edge_kernel<<<nblocks, nthreads, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_X, 1);
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], buff, gh->length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Injects an edge along X from a buffer into the GPU halo region.
 */
void inject_edge_X_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->recv_buff_d[i_buff];
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(buff, halo->recv_buff_h[i_buff], gh->length_X * sizeof(DataType), cudaMemcpyHostToDevice));
    }
    int const nthreads = 128; // number of threads per block
    int const nblocks = (gh->length_X + nthreads - 1) / nthreads; // total blocks needed
    inject_edge_kernel<<<nblocks, nthreads, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_X, 1);
}

/**
 * @brief Extracts an edge along Y from the GPU halo region into a buffer.
 */
void extract_edge_Y_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    DataType *buff = halo->send_buff_d[i_buff];
    int const nthreads = 128; // number of threads per block
    int const nblocks = (gh->length_Y + nthreads - 1) / nthreads; // total blocks needed
    extract_edge_kernel<<<nblocks, nthreads, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_Y, gh->dimx);
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], buff, gh->length_Y * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Injects an edge along Y from a buffer into the GPU halo region.
 */
void inject_edge_Y_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->recv_buff_d[i_buff];
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(buff, halo->recv_buff_h[i_buff], gh->length_Y * sizeof(DataType), cudaMemcpyHostToDevice));
    }
    int const nthreads = 128; // number of threads per block
    int const nblocks = (gh->length_Y + nthreads - 1) / nthreads; // total blocks needed
    inject_edge_kernel<<<nblocks, nthreads, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_Y, gh->dimx);
}

/**
 * @brief Extracts an edge along Z from the GPU halo region into a buffer.
 */
void extract_edge_Z_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    DataType *buff = halo->send_buff_d[i_buff];
    int const nthreads = 128; // number of threads per block
    int const nblocks = (gh->length_Z + nthreads - 1) / nthreads; // total blocks needed
    extract_edge_kernel<<<nblocks, nthreads, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_Z, gh->dimx * gh->dimy);
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], buff, gh->length_Z * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Injects an edge along Z from a buffer into the GPU halo region.
 */
void inject_edge_Z_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;

    DataType *buff = halo->recv_buff_d[i_buff];
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(buff, halo->recv_buff_h[i_buff], gh->length_Z * sizeof(DataType), cudaMemcpyHostToDevice));
    }
    int const nthreads = 128; // number of threads per block
    int const nblocks = (gh->length_Z + nthreads - 1) / nthreads; // total blocks needed
    inject_edge_kernel<<<nblocks, nthreads, 0, *(halo->streams[i_buff])>>>(x_d, buff, gh->length_Z, gh->dimx * gh->dimy);
}

/**
 * @brief Extracts a corner value from the GPU halo region into a buffer.
 * @note Only works for corner being a single element.
 */
void extract_corner_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(halo->send_buff_h[i_buff], x_d, sizeof(DataType), cudaMemcpyDeviceToHost));
    } else {
        CHECK_CUDA(cudaMemcpyAsync(
            halo->send_buff_d[i_buff],
            x_d,
            sizeof(DataType),
            cudaMemcpyDeviceToDevice,
            *(halo->streams[i_buff])
        ));
    }
}

/**
 * @brief Injects a corner value from a buffer into the GPU halo region.
 * @note Only works for corner being a single element.
 */
void inject_corner_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff)
{
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    if (host_buff) {
        CHECK_CUDA(cudaMemcpy(x_d, halo->recv_buff_h[i_buff], sizeof(DataType), cudaMemcpyHostToDevice));
    } else {
        CHECK_CUDA(cudaMemcpyAsync(
            x_d,
            halo->recv_buff_d[i_buff],
            sizeof(DataType),
            cudaMemcpyDeviceToDevice,
            *(halo->streams[i_buff])
        ));
    }
}

/**
 * @brief Sends the data region of halo to the specified rank.
 *        Can be used to gather the result at one process.
 */
void SendResult(int rank_recv, Halo *x_d, Problem *problem)
{
    DataType *send_addr_d = x_d->interior;
    DataType *send_buf_h = (DataType*) malloc(problem->nx * sizeof(DataType));
    for(int i = 0; i < problem->nz; i++){
        for(int j = 0; j<problem->ny; j++){
            CHECK_CUDA(cudaMemcpy(send_buf_h, send_addr_d, problem->nx * sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Send(send_buf_h, problem->nx, MPIDataType, rank_recv, 0, MPI_COMM_WORLD);
            send_addr_d += x_d->dimx;
        }
        send_addr_d += 2 * x_d->dimx;
    }
    free(send_buf_h);
}

/**
 * @brief Collects the data region of a halo from all processes in the correct order and stores it into result_h.
 *        Includes the own computation.
 */
void GatherResult(Halo *x_d, Problem *problem, DataType *result_h)
{
    DataType *own_data_d = x_d->interior;
    for(int i = 0; i<problem->gnz; i++){ // go through all gnz layers
        int pz_recv = i / problem->nz;
        for(int j = 0; j<problem->gny; j++){ // go through all gny rows
            int py_recv = j / problem->ny;
            for(int l = 0; l<problem->npx; l++){ // go thorugh all nx columns
                int px_recv = l;
                int rank_recv = pz_recv * problem->npx * problem->npy + py_recv * problem->npx + px_recv;
                if(px_recv == problem->px && py_recv == problem->py && pz_recv == problem->pz){ //gathering rank holds the data
                    CHECK_CUDA(cudaMemcpy(result_h, own_data_d, problem->nx * sizeof(DataType), cudaMemcpyDeviceToHost));
                    for(int k = 0; k<problem->nx; k++){
                    }
                    own_data_d += x_d->dimx;
                }else{
                    MPI_Recv(result_h, problem->nx, MPIDataType, rank_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                result_h += problem->nx;
            }
        }
        if(pz_recv == problem->pz){
            own_data_d += 2 * x_d->dimx;
        }
    }
}

/**
 * @brief Prints the data region of a halo.
 */
void PrintHalo(Halo *x_d)
{
    DataType *x_h = (DataType*) malloc(x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(x_h, x_d->x_d, x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType), cudaMemcpyDeviceToHost));
    for(int i = 0; i < x_d->dimz; i++){
        for(int j = 0; j < x_d->dimy; j++){
            for(int k = 0; k < x_d->dimx; k++){
                if(k == x_d->dimx - 1){
                    printf("\t");
                }
                printf("%f ", x_h[i*x_d->dimy*x_d->dimx + j*x_d->dimx + k]);
                if(k == 0){
                    printf("\t");
                }
            }
            if(j==0 || j == x_d->dimy - 2){
                printf("\n");
            }
            printf("\n");
        }
        printf("---\n");
    }
    free(x_h);
}

/**
 * @brief Generates a striped partial matrix on CPU.
 */
void GenerateStripedPartialMatrix(Problem *problem, DataType *A)
{
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    global_int_t gnx = problem->gnx; //global number of points in x
    global_int_t gny = problem->gny; //global number of points in y
    global_int_t gnz = problem->gnz; //global number of points in z
    
    for(int iz = 0; iz < nz; iz++){
        for(int iy = 0; iy < ny; iy++){
            for(int ix = 0; ix < nx; ix++){

                int gx = problem->gx0 + ix; //global x index
                int gy = problem->gy0 + iy; //global y index
                int gz = problem->gz0 + iz; //global z index
                
                for (int sz = -1; sz < 2; sz++){
                    for(int sy = -1; sy < 2; sy++){
                        for(int sx = -1; sx < 2; sx++){

                            if(gx + sx < 0 || gx + sx >= gnx ||
                                gy + sy < 0 || gy + sy >= gny ||
                                gz + sz < 0 || gz + sz >= gnz) {
                                    *A = 0.0;
                                    A++;
                                } else {
                                    if(sx == 0 && sy == 0 && sz == 0){
                                        *A = 26.0;
                                        A++;
                                    } else {
                                        *A = -1.0;
                                        A++;
                                    }
                                }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Verifies correctness of a striped partial matrix.
 */
bool VerifyPartialMatrix(DataType *striped_A_local_h, DataType *striped_A_global_h, int num_stripes, Problem *problem)
{
    for(int i = 0; i<problem->nz; i++){
        int gi0 = problem->gi0 + i * problem->gnx * problem->gny;
        for(int j = 0; j<problem->ny; j++){
            for(int k = 0; k<problem->nx; k++){
                for(int l = 0; l<num_stripes; l++){
                    if(striped_A_local_h[(k + j*problem->nx + i*problem->nx*problem->ny)*num_stripes + l] != striped_A_global_h[(gi0 + j*problem->gnx + k)*num_stripes + l]){
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the data region of a halo is zero.
 */
bool IsHaloZero(Halo *x_d)
{
    DataType *x_h = (DataType*) malloc(x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(x_h, x_d->x_d, x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType), cudaMemcpyDeviceToHost));
    //check front and back
    for(int i = 0; i < x_d->dimx * x_d->dimy; i++){
        if(x_h[i] != 0.0 || x_h[i + x_d->dimx * x_d->dimy * (x_d->dimz - 1)] != 0.0){
            return false;
        }
    }
    //check middle part
    x_h += x_d->dimx * x_d->dimy;
    for(int iz = 0; iz < x_d->dimz-2; iz++){
        for(int iy = 0; iy < x_d->dimy; iy++){
            for(int ix = 0; ix < x_d->dimx; ix++){
                if(ix == 0 || ix == x_d->dimx - 1){
                    if(x_h[ix + iy * x_d->dimx + iz * x_d->dimx * x_d->dimy] != 0.0){
                        printf("ix = %d, iy = %d, iz = %d dimx=%d, dimy=%d, dimz=%d\n", ix, iy, iz, x_d->dimx, x_d->dimy, x_d->dimz);
                        return false;
                    }
                }
                if(iy == 0 || iy == x_d->dimy - 1){
                    if(x_h[ix + iy * x_d->dimx + iz * x_d->dimx * x_d->dimy] != 0.0){
                        printf("ix = %d, iy = %d, iz = %d dimx=%d, dimy=%d, dimz=%d\n", ix, iy, iz, x_d->dimx, x_d->dimy, x_d->dimz);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

__inline__ __device__ local_int_t global_i_to_halo_i(
    global_int_t i,
    local_int_t nx, local_int_t ny, local_int_t nz,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
    )
    {
        return ((i % gnx) - px * nx + 1) +
            ((((i / gnx) % gny) - py * ny + 1) * (nx + 2)) +
            (((i / (gnx * gny)) - pz * nz + 1) * ((nx + 2) * (ny + 2))); //should be equivalent to the above
}


__global__ void extract_COO_data_kernel(DataType *x_d, global_int_t total_to_send, global_int_t *idx_to_extract, DataType *buff, local_int_t nx, local_int_t ny, local_int_t nz, local_int_t dimx, local_int_t dimy,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0, int px, int py, int pz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_to_send) return;
    global_int_t i = idx_to_extract[tid];
    // Convert global index to local index
    local_int_t local_i = global_i_to_halo_i(i, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
    // Convert halo local to halo index
    global_int_t halo_i = local_i_to_halo_i(local_i, nx, ny, nz, dimx, dimy);
    // Extract the value from the halo
    DataType value = x_d[halo_i];
    // Store the value in the buffer
    buff[tid] = value;
}

void extract_COO_data(Halo *halo, Problem *p, striped_partial_Matrix<DataType> &A_local){
    global_int_t total_to_send = A_local.total_to_send_COO;
    int num_threads = 512; // number of threads per block
    int num_blocks = (total_to_send + num_threads - 1) / num_threads;
    // Launch the kernel to extract COO data from the halo
    extract_COO_data_kernel<<<num_blocks, num_threads, 0, *(halo->streams[28])>>>(halo->x_d, total_to_send, A_local.ptr_idx_to_send_COO_h[0], A_local.ptr_send_buff_COO_h[0], p->nx, p->ny, p->nz, halo->dimx, halo->dimy, p->gnx, p->gny, p->gnz, p->gi0, p->px, p->py, p->pz);
    CHECK_CUDA(cudaMemcpyAsync(A_local.ptr_recv_buff_COO_h[p->rank], A_local.ptr_send_buff_COO_h[p->rank], A_local.req_per_rank_COO[p->rank] * sizeof(DataType), cudaMemcpyDeviceToDevice, *(halo->streams[28]))); //we also extracted the data needed locally. now, we place it in the recv buffer
}
