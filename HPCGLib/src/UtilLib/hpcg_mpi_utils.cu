#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <testing.hpp>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <cassert>
#include <stdbool.h>

__inline__ __device__ global_int_t local_i_to_halo_i(
    int i, 
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
    )
    {
        return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

__global__ void inject_data_to_halo_kernel(DataType *x_d, DataType *data, int nx, int ny, int nz, int dimx, int dimy){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x){
            int hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
            x_d[hi] = data[i];
    }
}

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem){
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

    void (*tmp_extraction_functions[NUMBER_NEIGHBORS])(Halo *halo, DataType *buff, GhostCell *gh) = {
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

    void (*tmp_injection_functions[NUMBER_NEIGHBORS])(Halo *halo, DataType *buff, GhostCell *gh) = {
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

void InitHaloMemGPU(Halo *halo, Problem *problem){
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    int dimx = nx + 2;
    int dimy = ny + 2;
    int dimz = nz + 2;
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
    for(int i = 0; i < NUMBER_NEIGHBORS; i++){
        CHECK_CUDA(cudaMalloc(&(halo->send_buff_d[i]), problem->count_exchange[i] * sizeof(DataType)));
        CHECK_CUDA(cudaMemset(halo->send_buff_d[i], 0, problem->count_exchange[i] * sizeof(DataType)));
        CHECK_CUDA(cudaMalloc(&(halo->recv_buff_d[i]), problem->count_exchange[i] * sizeof(DataType)));
        CHECK_CUDA(cudaMemset(halo->recv_buff_d[i], 0, problem->count_exchange[i] * sizeof(DataType)));
        
    }
    // Ensure all device memory setting is complete
    CHECK_CUDA(cudaDeviceSynchronize());
    
}

void InitHaloMemCPU(Halo *halo, Problem *problem){
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    int dimx = nx + 2;
    int dimy = ny + 2;
    int dimz = nz + 2;
    halo->nx = nx;
    halo->ny = ny;
    halo->nz = nz;
    halo->dimx = dimx;
    halo->dimy = dimy;
    halo->dimz = dimz;

    // allocate memory for send and receive buffers on CPU
    for(int i = 0; i < NUMBER_NEIGHBORS; i++){
        
        halo->send_buff_h[i] = (DataType *) malloc(problem->count_exchange[i] * sizeof(DataType));
        memset(halo->send_buff_h[i], 0, problem->count_exchange[i] * sizeof(DataType));
        halo->recv_buff_h[i] = (DataType *) malloc(problem->count_exchange[i] * sizeof(DataType));
        memset(halo->recv_buff_h[i], 0, problem->count_exchange[i] * sizeof(DataType));
        
    }
}

/*
* Initializes the memory for halo on both CPU and GPU and initializes all data memory with zeros.
*/
void InitHalo(Halo *halo, Problem *problem){
    halo->problem = problem;
    InitHaloMemGPU(halo, problem);
    InitHaloMemCPU(halo, problem);
    SetHaloZeroGPU(halo);
}

void SetHaloZeroGPU(Halo *halo){
    CHECK_CUDA(cudaMemset(halo->x_d, 0, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType)));
}

void InjectDataToHalo(Halo *halo, DataType *data){
    int n = halo->nx * halo->ny * halo->nz;
    int num_threads = 1024;
    int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(n, num_threads));
    inject_data_to_halo_kernel<<<num_blocks, num_threads>>>(halo->x_d, data, halo->nx, halo->ny, halo->nz, halo->dimx, halo->dimy);
}

/*
* Initialize the halo with global index
*/
void SetHaloGlobalIndexGPU(Halo *halo, Problem *problem){
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

/*
* Initialize the halo with 1.0/(global index + 1.0)
*/
void SetHaloQuotientGlobalIndexGPU(Halo *halo, Problem *problem){
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

/*
* Initialize the halo with random numbers between min and max
* Seed = input seed + rank for each process
*/
void SetHaloRandomGPU(Halo *halo, Problem *problem, int min, int max, int seed){
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

void FreeHaloGPU(Halo *halo){
    CHECK_CUDA(cudaFree(halo->x_d));
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
            CHECK_CUDA(cudaFree(halo->send_buff_d[i]));
            CHECK_CUDA(cudaFree(halo->recv_buff_d[i]));
    }
}

void FreeHaloCPU(Halo *halo){
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
            free(halo->send_buff_h[i]);
            free(halo->recv_buff_h[i]);
    }
}

void FreeHalo(Halo *halo){
    FreeHaloGPU(halo);
    FreeHaloCPU(halo);
}

void InitGPU(Problem *problem){
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount > 0);
    CHECK_CUDA(cudaSetDevice(problem->rank % deviceCount));
    //printf("Rank=%d:\t\t Set my device to device=%d, available=%d.\n", problem->rank, problem->rank % deviceCount, deviceCount);
}
/* x_d is the pointer to the first element in the slice */
/* every thread fills an element */
__global__ void extract_xz_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Z, int slice_X, int slice_Z){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (z_loc<length_Z) slice_d[tid]=x_d[z_loc*slice_Z + x_loc*slice_X];
}

__global__ void inject_xz_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Z, int slice_X, int slice_Z){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (z_loc<length_Z) x_d[z_loc*slice_Z + x_loc*slice_X]=slice_d[tid];
}

__global__ void extract_yz_plane_kernel(DataType *x_d, DataType *slice_d, int length_Y, int length_Z, int slice_Y, int slice_Z){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z_loc = tid / length_Y;
    int y_loc = tid % length_Y;
    if (z_loc<length_Z) slice_d[tid]=x_d[z_loc*slice_Z + y_loc*slice_Y];
}

__global__ void inject_yz_plane_kernel(DataType *x_d, DataType *slice_d, int length_Y, int length_Z, int slice_Y, int slice_Z){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z_loc = tid / length_Y;
    int y_loc = tid % length_Y;
    if (z_loc<length_Z) x_d[z_loc*slice_Z + y_loc*slice_Y]=slice_d[tid];
}
__global__ void extract_xy_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Y, int slice_X, int slice_Y){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int y_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (y_loc<length_Y) slice_d[tid]=x_d[y_loc*slice_Y + x_loc*slice_X];
}

__global__ void inject_xy_plane_kernel(DataType *x_d, DataType *slice_d, int length_X, int length_Y, int slice_X, int slice_Y){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int y_loc = tid / length_X;
    int x_loc = tid % length_X;
    if (y_loc<length_Y) x_d[y_loc*slice_Y + x_loc*slice_X]=slice_d[tid];
}

void extract_horizontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int i = 0; i < length_Z; i++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h += length_X;
        x_d += dimx * dimy;
    } */
    DataType *slice_d;
    CHECK_CUDA(cudaMalloc(&slice_d, length_X*length_Z*sizeof(DataType)));

    // collect halo on device
    int const nthread=256;
    int const nblock = (length_X*length_Z + nthread - 1) / nthread;
    extract_xz_plane_kernel<<<nblock,nthread>>>(x_d, slice_d, length_X, length_Z, 1, dimy*dimx);
    
    // copy from device to host
    CHECK_CUDA(cudaMemcpy(x_h, slice_d, length_X*length_Z*sizeof(DataType), cudaMemcpyDeviceToHost));

void extract_horizontal_plane_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int i = 0; i < gh->length_Z; i++) {
        CHECK_CUDA(cudaMemcpy(buff, x_d, gh->length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        buff += gh->length_X;
        x_d += gh->dimx * gh->dimy;
    }
}

void inject_horizontal_plane_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int i = 0; i < gh->length_Z; i++) {
        CHECK_CUDA(cudaMemcpy(x_d, buff, gh->length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += gh->dimx * gh->dimy;
        buff += gh->length_X;
    }
void inject_horizontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int i = 0; i < length_Z; i++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx * dimy;
        x_h += length_X;
    } */

    // copy halo to gpu, slice_d
    DataType *slice_d;
    CHECK_CUDA(cudaMalloc(&slice_d, length_X*length_Z*sizeof(DataType)));
    CHECK_CUDA(cudaMemcpy(slice_d, x_h, length_X*length_Z*sizeof(DataType), cudaMemcpyHostToDevice));
    
    // fill x_d with slice_d
    int const nthread=256;
    int const nblock = (length_X*length_Z + nthread - 1) / nthread;
    inject_xz_plane_kernel<<<nblock,nthread>>>(x_d, slice_d, length_X, length_Z, 1, dimy*dimx);

}

void extract_vertical_plane_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int i = 0; i < gh->length_Z; i++) {
        for (int j = 0; j < gh->length_Y; j++) {
            CHECK_CUDA(cudaMemcpy(buff, x_d, sizeof(DataType), cudaMemcpyDeviceToHost));
            buff++;
            x_d += gh->dimx;
void extract_vertical_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* 
    for(int i = 0; i < length_Z; i++){
        for(int j = 0; j < length_Y; j++){
            CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
            x_h++;
            x_d += dimx;
        }
        x_d += dimx * (dimy - length_Y);
    } 
    */
    // allocate halo on device
    DataType *slice_d;
    CHECK_CUDA(cudaMalloc(&slice_d, length_Y*length_Z*sizeof(DataType)));

    // collect halo on device
    int nthread=256;
    int nblock = (length_Y*length_Z + nthread - 1) / nthread;
    extract_yz_plane_kernel<<<nblock,nthread>>>(x_d, slice_d, length_Y, length_Z, dimx, dimy*dimx);
    
    // copy from device to host
    CHECK_CUDA(cudaMemcpy(x_h, slice_d, length_Y*length_Z*sizeof(DataType), cudaMemcpyDeviceToHost));
        x_d += gh->dimx * (gh->dimy - gh->length_Y);
    }
}

void inject_vertical_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int i = 0; i < length_Z; i++){
        for(int j = 0; j < length_Y; j++){
            CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
            x_d += dimx;
            x_h ++;
void inject_vertical_plane_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int i = 0; i < gh->length_Z; i++) {
        for (int j = 0; j < gh->length_Y; j++) {
            CHECK_CUDA(cudaMemcpy(x_d, buff, sizeof(DataType), cudaMemcpyHostToDevice));
            x_d += gh->dimx;
            buff++;
        }
        x_d += dimx * (dimy - length_Y);
    } */

    // copy host x_h to gpu, slice_d
    DataType *slice_d;
    CHECK_CUDA(cudaMalloc(&slice_d, length_Y*length_Z*sizeof(DataType)));
    CHECK_CUDA(cudaMemcpy(slice_d, x_h, length_Y*length_Z*sizeof(DataType), cudaMemcpyHostToDevice));
    
    // fill x_d with slice_d
    int const nthread=256;
    int const nblock = (length_Y*length_Z + nthread - 1) / nthread;
    inject_yz_plane_kernel<<<nblock,nthread>>>(x_d, slice_d, length_Y, length_Z, dimx, dimy*dimx);

        x_d += gh->dimx * (gh->dimy - gh->length_Y);
    }
}

void extract_frontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int i = 0; i < length_Y; i++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h += length_X;
        x_d += dimx;
    } */

    // allocate halo on device
    DataType *slice_d;
    CHECK_CUDA(cudaMalloc(&slice_d, length_X*length_Y*sizeof(DataType)));

    // collect halo on device
    int nthread=256;
    int nblock = (length_X*length_Y + nthread - 1) / nthread;
    extract_xy_plane_kernel<<<nblock,nthread>>>(x_d, slice_d, length_X, length_Y, 1, dimx);
    
    // copy from device to host
    CHECK_CUDA(cudaMemcpy(x_h, slice_d, length_X*length_Y*sizeof(DataType), cudaMemcpyDeviceToHost));

void extract_frontal_plane_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int i = 0; i < gh->length_Y; i++) {
        CHECK_CUDA(cudaMemcpy(buff, x_d, gh->length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        buff += gh->length_X;
        x_d += gh->dimx;
    }
}

void inject_frontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int i = 0; i < length_Y; i++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx;
        x_h += length_X;
    } */

    // copy host x_h to gpu, slice_d
    DataType *slice_d;
    CHECK_CUDA(cudaMalloc(&slice_d, length_X*length_Y*sizeof(DataType)));
    CHECK_CUDA(cudaMemcpy(slice_d, x_h, length_X*length_Y*sizeof(DataType), cudaMemcpyHostToDevice));
    
    // fill x_d with slice_d
    int const nthread=256;
    int const nblock = (length_X*length_Y + nthread - 1) / nthread;
    inject_xy_plane_kernel<<<nblock,nthread>>>(x_d, slice_d, length_X, length_Y, 1, dimx);


}

__global__ void extract_edge_kernel(DataType *x_d, DataType *slice_d, int length_X, int slice_X){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid<length_X) slice_d[tid]=x_d[tid*slice_X];
}

__global__ void inject_edge_kernel(DataType *x_d, DataType *slice_d, int length_X, int slice_X){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid<length_X) x_d[tid*slice_X] = slice_d[tid];
void inject_frontal_plane_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int i = 0; i < gh->length_Y; i++) {
        CHECK_CUDA(cudaMemcpy(x_d, buff, gh->length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += gh->dimx;
        buff += gh->length_X;
    }
}

void extract_edge_X_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    CHECK_CUDA(cudaMemcpy(buff, x_d, gh->length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
}

void inject_edge_X_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    CHECK_CUDA(cudaMemcpy(x_d, buff, gh->length_X * sizeof(DataType), cudaMemcpyHostToDevice));
}

void extract_edge_Y_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int j = 0; j < length_Y; j++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h++;
        x_d += dimx;
    } */
    DataType *arr_d;
    CHECK_CUDA(cudaMalloc(&arr_d, length_Y * sizeof(DataType)));

    int const nthreads = 128;
    int const nblocks = (length_Y + nthreads - 1)/nthreads;
    extract_edge_kernel<<<nblocks, nthreads>>>(x_d, arr_d, length_Y, dimx);

    CHECK_CUDA(cudaMemcpy(x_h, arr_d, length_Y * sizeof(DataType), cudaMemcpyDeviceToHost));
void extract_edge_Y_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int j = 0; j < gh->length_Y; j++) {
        CHECK_CUDA(cudaMemcpy(buff, x_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        buff++;
        x_d += gh->dimx;
    }
}

void inject_edge_Y_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int j = 0; j < gh->length_Y; j++) {
        CHECK_CUDA(cudaMemcpy(x_d, buff, sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += gh->dimx;
        buff++;
    }
void inject_edge_Y_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int j = 0; j < length_Y; j++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx;
        x_h ++;
    } */

    DataType *arr_d;
    CHECK_CUDA(cudaMalloc(&arr_d, length_Y * sizeof(DataType)));
    CHECK_CUDA(cudaMemcpy(arr_d, x_h, length_Y * sizeof(DataType), cudaMemcpyHostToDevice));

    int const nthreads = 128;
    int const nblocks = (length_Y + nthreads - 1)/nthreads;
    inject_edge_kernel<<<nblocks, nthreads>>>(x_d, arr_d, length_Y, dimx);
}

void extract_edge_Z_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int j = 0; j < length_Z; j++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h++;
        x_d += dimx * dimy;
    } */

    DataType *arr_d;
    CHECK_CUDA(cudaMalloc(&arr_d, length_Z * sizeof(DataType)));

    int const nthreads = 128;
    int const nblocks = (length_Z + nthreads - 1)/nthreads;
    extract_edge_kernel<<<nblocks, nthreads>>>(x_d, arr_d, length_Z, dimx*dimy);

    CHECK_CUDA(cudaMemcpy(x_h, arr_d, length_Z * sizeof(DataType), cudaMemcpyDeviceToHost));

void extract_edge_Z_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int j = 0; j < gh->length_Z; j++) {
        CHECK_CUDA(cudaMemcpy(buff, x_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        buff++;
        x_d += gh->dimx * gh->dimy;
    }
}

void inject_edge_Z_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int j = 0; j < gh->length_Z; j++) {
        CHECK_CUDA(cudaMemcpy(x_d, buff, sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += gh->dimx * gh->dimy;
        buff++;
    }
}

void extract_corner_from_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int z = 0; z < gh->length_Z; z++) {
        for (int y = 0; y < gh->length_Y; y++) {
            // Copy a contiguous block of length_X elements (one row of the corner)
            CHECK_CUDA(cudaMemcpy(buff, x_d, gh->length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
            buff += gh->length_X;
            x_d += gh->dimx;  // move to the next row in the halo
        }
        // Jump to the start of the next z-plane:
        x_d += gh->dimx * (gh->dimy - gh->length_Y);
    }
}

void inject_corner_to_GPU(Halo *halo, DataType *buff, GhostCell *gh) {
    local_int_t k = gh->x + gh->y * gh->dimx + gh->z * gh->dimx * gh->dimy;
    DataType *x_d = halo->x_d + k;
    for (int z = 0; z < gh->length_Z; z++) {
        for (int y = 0; y < gh->length_Y; y++) {
            // Copy a contiguous block of length_X elements (one row of the corner)
            CHECK_CUDA(cudaMemcpy(x_d, buff, gh->length_X * sizeof(DataType), cudaMemcpyHostToDevice));
            buff += gh->length_X;
            x_d += gh->dimx;  // move to the next row in the halo
        }
        // Jump to the start of the next z-plane:
        x_d += gh->dimx * (gh->dimy - gh->length_Y);
    }
void inject_edge_Z_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    /* for(int j = 0; j < length_Z; j++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx * dimy;
        x_h ++;
    } */

    DataType *arr_d;
    CHECK_CUDA(cudaMalloc(&arr_d, length_Z * sizeof(DataType)));
    CHECK_CUDA(cudaMemcpy(arr_d, x_h, length_Z * sizeof(DataType), cudaMemcpyHostToDevice));

    int const nthreads = 128;
    int const nblocks = (length_Z + nthreads - 1)/nthreads;
    inject_edge_kernel<<<nblocks, nthreads>>>(x_d, arr_d, length_Z, dimx*dimy);

}

//correctness verified
void SendResult(int rank_recv, Halo *x_d, Problem *problem){
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

//correctness verified
void GatherResult(Halo *x_d, Problem *problem, DataType *result_h){
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

void PrintHalo(Halo *x_d){
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

void GenerateStripedPartialMatrix(Problem *problem, DataType *A){
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

bool VerifyPartialMatrix(DataType *striped_A_local_h, DataType *striped_A_global_h, int num_stripes, Problem *problem){
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

bool IsHaloZero(Halo *x_d){
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
