#include "UtilLib/hpcg_mpi_utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <testing.hpp>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <cassert>
#include <stdbool.h>

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem){
    problem->npx = npx; //number of processes in x
    problem->npy = npy; //number of processes in y
    problem->npz = npz; //number of processes in z
    problem->nx = nx; //number of grid points of processes subdomain in x
    problem->ny = ny; //number of grid points of processes subdomain in y
    problem->nz = nz; //number of grid points of processes subdomain in z
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
}

void InitHaloMemGPU(Halo *halo, int nx, int ny, int nz){
    int dimx = nx + 2;
    int dimy = ny + 2;
    int dimz = nz + 2;
    DataType *x_d;
    CHECK_CUDA(cudaMalloc(&x_d, dimx * dimy * dimz * sizeof(DataType)));
    halo->x_d = x_d;
    DataType *interior = x_d + dimx * dimy + dimx + 1;
    halo->nx = nx;
    halo->ny = ny;
    halo->nz = nz;
    halo->dimx = dimx;
    halo->dimy = dimy;
    halo->dimz = dimz;
    halo->interior = interior;
}

void SetHaloZeroGPU(Halo *halo){
    CHECK_CUDA(cudaMemset(halo->x_d, 0, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType)));
}

//correctness verified
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

void FreeHaloGPU(Halo *halo){
    CHECK_CUDA(cudaFree(halo->x_d));
}

void InitGPU(Problem *problem){
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount > 0);
    CHECK_CUDA(cudaSetDevice(problem->rank % deviceCount));
    //printf("Rank=%d:\t\t Set my device to device=%d, available=%d.\n", problem->rank, problem->rank % deviceCount, deviceCount);
}

//copies to host and back to device, no device-to-device copy
//TODO: Replace malloc per exchange with malloc once at beginning
void ExchangeHalo(Halo *halo, Problem *problem){
    MPI_Barrier(MPI_COMM_WORLD);
    int dimx = halo->dimx;
    int dimy = halo->dimy;
    int dimz = halo->dimz;
    int nx = halo->nx;
    int ny = halo->ny;
    int nz = halo->nz;
    DataType *x_d = halo->x_d;
    //exchange north if got north
    if(problem->py > 0){
        DataType *north_send = (DataType*) malloc(nx * nz * sizeof(DataType));
        extract_horizontal_plane_from_GPU(x_d, north_send, 1, 1, 1, nx, nz, dimx, dimy, dimz);
        DataType *north_receive = (DataType*) malloc(nx * nz * sizeof(DataType));
        MPI_Sendrecv(north_send, nx * nz, MPIDataType, problem->rank - problem->npx, NORTH, north_receive, nx * nz, MPIDataType, problem->rank - problem->npx, SOUTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_horizontal_plane_to_GPU(x_d, north_receive, 1, 0, 1, nx, nz, dimx, dimy, dimz);
        free(north_send);
        free(north_receive);
    }
    
    //exchange east if got east
    if(problem->px < problem->npx - 1){
        DataType *east_send = (DataType*) malloc(ny * nz * sizeof(DataType));
        extract_vertical_plane_from_GPU(x_d, east_send, dimx - 2, 1, 1, ny, nz, dimx, dimy, dimz);
        DataType *east_receive = (DataType*) malloc(ny * nz * sizeof(DataType));
        MPI_Sendrecv(east_send, ny * nz, MPIDataType, problem->rank + 1, EAST, east_receive, ny * nz, MPIDataType, problem->rank + 1, WEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_vertical_plane_to_GPU(x_d, east_receive, dimx - 1, 1, 1, ny, nz, dimx, dimy, dimz);
        free(east_send);
        free(east_receive);
        
    }
    
    //exchange south if got south
    if(problem->py < problem->npy - 1){
        DataType *south_send = (DataType*) malloc(nx * nz * sizeof(DataType));
        extract_horizontal_plane_from_GPU(x_d, south_send, 1, dimy - 2, 1, nx, nz, dimx, dimy, dimz);
        DataType *south_receive = (DataType*) malloc(nx * nz * sizeof(DataType));
        MPI_Sendrecv(south_send, nx * nz, MPIDataType, problem->rank + problem->npx, SOUTH, south_receive, nx * nz, MPIDataType, problem->rank + problem->npx, NORTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_horizontal_plane_to_GPU(x_d, south_receive, 1, dimy - 1, 1, nx, nz, dimx, dimy, dimz);
        free(south_send);
        free(south_receive);
    }
    
    //exchange west if got west
    if(problem->px > 0){
        DataType *west_send = (DataType*) malloc(ny * nz * sizeof(DataType));
        extract_vertical_plane_from_GPU(x_d, west_send, 1, 1, 1, ny, nz, dimx, dimy, dimz);
        DataType *west_receive = (DataType*) malloc(ny * nz * sizeof(DataType));
        MPI_Sendrecv(west_send, ny * nz, MPIDataType, problem->rank - 1, WEST, west_receive, ny * nz, MPIDataType, problem->rank - 1, EAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_vertical_plane_to_GPU(x_d, west_receive, 0, 1, 1, ny, nz, dimx, dimy, dimz);
        free(west_send);
        free(west_receive);
    }
    //exchange ne if got ne
    if(problem->px < problem->npx - 1 && problem->py > 0){
        DataType *ne_send = (DataType*) malloc(nz * sizeof(DataType));
        extract_edge_Z_from_GPU(x_d, ne_send, dimx - 2, 1, 1, nz, dimx, dimy, dimz);
        DataType *ne_receive = (DataType*) malloc(nz * sizeof(DataType));
        MPI_Sendrecv(ne_send, nz, MPIDataType, problem->rank - problem->npx + 1, NE, ne_receive, nz, MPIDataType, problem->rank - problem->npx + 1, SW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, ne_receive, dimx - 1, 0, 1, nz, dimx, dimy, dimz);
        free(ne_send);
        free(ne_receive);
    }
    
    //exchange se if got se
    if(problem->py < problem->npy - 1 && problem->px < problem->npx - 1){
        DataType *se_send = (DataType*) malloc(nz * sizeof(DataType));
        extract_edge_Z_from_GPU(x_d, se_send, dimx - 2, dimy - 2, 1, nz, dimx, dimy, dimz);
        DataType *se_receive = (DataType*) malloc(nz * sizeof(DataType));
        MPI_Sendrecv(se_send, nz, MPIDataType, problem->rank + problem->npx + 1, SE, se_receive, nz, MPIDataType, problem->rank + problem->npx + 1, NW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, se_receive, dimx - 1, dimy - 1, 1, nz, dimx, dimy, dimz);
        free(se_send);
        free(se_receive);
    }
    
    //exchange sw if got sw
    if(problem->px > 0 && problem->py < problem->npy - 1){
        DataType *sw_send = (DataType*) malloc(nz * sizeof(DataType));
        extract_edge_Z_from_GPU(x_d, sw_send, 1, dimy - 2, 1, nz, dimx, dimy, dimz);
        DataType *sw_receive = (DataType*) malloc(nz * sizeof(DataType));
        MPI_Sendrecv(sw_send, nz, MPIDataType, problem->rank + problem->npx - 1, SW, sw_receive, nz, MPIDataType, problem->rank + problem->npx - 1, NE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, sw_receive, 0, dimy - 1, 1, nz, dimx, dimy, dimz);
        free(sw_send);
        free(sw_receive);
    }
    
    //exchange nw if got nw
    if(problem->py > 0 && problem->px > 0){
        DataType *nw_send = (DataType*) malloc(nz * sizeof(DataType));
        extract_edge_Z_from_GPU(x_d, nw_send, 1, 1, 1, nz, dimx, dimy, dimz);
        DataType *nw_receive = (DataType*) malloc(nz * sizeof(DataType));
        MPI_Sendrecv(nw_send, nz, MPIDataType, problem->rank - problem->npx - 1, NW, nw_receive, nz, MPIDataType, problem->rank - problem->npx - 1, SE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, nw_receive, 0, 0, 1, nz, dimx, dimy, dimz);
        free(nw_send);
        free(nw_receive);
    }
    
    //exchange front if got front
    if(problem->pz > 0){
        DataType *front_send = (DataType*) malloc(nx * ny * sizeof(DataType));
        extract_frontal_plane_from_GPU(x_d, front_send, 1, 1, 1, nx, ny, dimx, dimy, dimz);
        DataType *front_receive = (DataType*) malloc(nx * ny * sizeof(DataType));
        MPI_Sendrecv(front_send, nx * ny, MPIDataType, problem->rank - problem->npx * problem->npy, FRONT, front_receive, nx * ny, MPIDataType, problem->rank - problem->npx * problem->npy, BACK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_frontal_plane_to_GPU(x_d, front_receive, 1, 1, 0, nx, ny, dimx, dimy, dimz);
        free(front_send);
        free(front_receive);
    }
    //exchange back if got back
    if(problem->pz < problem->npz-1){
        DataType *back_send = (DataType*) malloc(nx * ny * sizeof(DataType));
        extract_frontal_plane_from_GPU(x_d, back_send, 1, 1, dimz-2, nx, ny, dimx, dimy, dimz);
        DataType *back_receive = (DataType*) malloc(nx * ny * sizeof(DataType));
        MPI_Sendrecv(back_send, nx * ny, MPIDataType, problem->rank + problem->npx * problem->npy, BACK, back_receive, nx * ny, MPIDataType, problem->rank + problem->npx * problem->npy, FRONT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_frontal_plane_to_GPU(x_d, back_receive, 1, 1, dimz-1, nx, ny, dimx, dimy, dimz);
        free(back_send);
        free(back_receive);
    }
    //exchange front_north if got front_north
    if(problem->py > 0 && problem->pz > 0){
        DataType *front_north_send = (DataType*) malloc(nx * sizeof(DataType));
        extract_edge_X_from_GPU(x_d, front_north_send, 1, 1, 1, nx, dimx, dimy, dimz);
        DataType *front_north_receive = (DataType*) malloc(nx * sizeof(DataType));
        MPI_Sendrecv(front_north_send, nx, MPIDataType, problem->rank - problem->npx * (problem->npy+1), FRONT_NORTH, front_north_receive, nx, MPIDataType, problem->rank - problem->npx * (problem->npy+1), BACK_SOUTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, front_north_receive, 1, 0, 0, nx, dimx, dimy, dimz);
        free(front_north_send);
        free(front_north_receive);
    }
    //exchange front_east if got front_east
    if(problem->px < problem->npx - 1 && problem->pz > 0){
        DataType *front_east_send = (DataType*) malloc(ny * sizeof(DataType));
        extract_edge_Y_from_GPU(x_d, front_east_send, dimx-2, 1, 1, ny, dimx, dimy, dimz);
        DataType *front_east_receive = (DataType*) malloc(ny * sizeof(DataType));
        MPI_Sendrecv(front_east_send, ny, MPIDataType, problem->rank - problem->npx * problem->npy + 1, FRONT_EAST, front_east_receive, ny, MPIDataType, problem->rank - problem->npx * problem->npy + 1, BACK_WEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, front_east_receive, dimx-1, 1, 0, ny, dimx, dimy, dimz);
        free(front_east_send);
        free(front_east_receive);
    }
    //exchange front_south if got front_south
    if(problem->py < problem->npy - 1 && problem->pz > 0){
        DataType *front_south_send = (DataType*) malloc(nx * sizeof(DataType));
        extract_edge_X_from_GPU(x_d, front_south_send, 1, dimy - 2, 1, nx, dimx, dimy, dimz);
        DataType *front_south_receive = (DataType*) malloc(nx * sizeof(DataType));
        MPI_Sendrecv(front_south_send, nx, MPIDataType, problem->rank - problem->npx * (problem->npy-1), FRONT_SOUTH, front_south_receive, nx, MPIDataType, problem->rank - problem->npx * (problem->npy-1), BACK_NORTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, front_south_receive, 1, dimy - 1, 0, nx, dimx, dimy, dimz);
        free(front_south_send);
        free(front_south_receive);
    }
    //exchange front_west if got front_west
    if(problem->px > 0 && problem->pz > 0){
        DataType *front_west_send = (DataType*) malloc(ny * sizeof(DataType));
        extract_edge_Y_from_GPU(x_d, front_west_send, 1, 1, 1, ny, dimx, dimy, dimz);
        DataType *front_west_receive = (DataType*) malloc(ny * sizeof(DataType));
        MPI_Sendrecv(front_west_send, ny, MPIDataType, problem->rank - problem->npx * problem->npy - 1, FRONT_WEST, front_west_receive, ny, MPIDataType, problem->rank - problem->npx * problem->npy - 1, BACK_EAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, front_west_receive, 0, 1, 0, ny, dimx, dimy, dimz);
        free(front_west_send);
        free(front_west_receive);
    }
    //exchange back_north if got back_north
    if(problem->py > 0 && problem->pz < problem->npz-1){
        DataType *back_north_send = (DataType*) malloc(nx * sizeof(DataType));
        extract_edge_X_from_GPU(x_d, back_north_send, 1, 1, dimz-2, nx, dimx, dimy, dimz);
        DataType *back_north_receive = (DataType*) malloc(nx * sizeof(DataType));
        MPI_Sendrecv(back_north_send, nx, MPIDataType, problem->rank + problem->npx * (problem->npy-1), BACK_NORTH, back_north_receive, nx, MPIDataType, problem->rank + problem->npx * (problem->npy-1), FRONT_SOUTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, back_north_receive, 1, 0, dimz-1, nx, dimx, dimy, dimz);
        free(back_north_send);
        free(back_north_receive);
    }
    //exchange back_east if got back_east
    if(problem->px < problem->npx - 1 && problem->pz < problem->npz-1){
        DataType *back_east_send = (DataType*) malloc(ny * sizeof(DataType));
        extract_edge_Y_from_GPU(x_d, back_east_send, dimx-2, 1, dimz-2, ny, dimx, dimy, dimz);
        DataType *back_east_receive = (DataType*) malloc(ny * sizeof(DataType));
        MPI_Sendrecv(back_east_send, ny, MPIDataType, problem->rank + problem->npx * problem->npy + 1, BACK_EAST, back_east_receive, ny, MPIDataType, problem->rank + problem->npx * problem->npy + 1, FRONT_WEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, back_east_receive, dimx-1, 1, dimz-1, ny, dimx, dimy, dimz);
        free(back_east_send);
        free(back_east_receive);
    }
    //exchange back_south if got back_south
    if(problem->py < problem->npy - 1 && problem->pz < problem->npz-1){
        DataType *back_south_send = (DataType*) malloc(nx * sizeof(DataType));
        extract_edge_X_from_GPU(x_d, back_south_send, 1, dimy - 2, dimz-2, nx, dimx, dimy, dimz);
        DataType *back_south_receive = (DataType*) malloc(nx * sizeof(DataType));
        MPI_Sendrecv(back_south_send, nx, MPIDataType, problem->rank + problem->npx * (problem->npy+1), BACK_SOUTH, back_south_receive, nx, MPIDataType, problem->rank + problem->npx * (problem->npy+1), FRONT_NORTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, back_south_receive, 1, dimy - 1, dimz-1, nx, dimx, dimy, dimz);
        free(back_south_send);
        free(back_south_receive);
    }
    //exchange back_west if got back_west
    if(problem->px > 0 && problem->pz < problem->npz-1){
        DataType *back_west_send = (DataType*) malloc(ny * sizeof(DataType));
        extract_edge_Y_from_GPU(x_d, back_west_send, 1, 1, dimz-2, ny, dimx, dimy, dimz);
        DataType *back_west_receive = (DataType*) malloc(ny * sizeof(DataType));
        MPI_Sendrecv(back_west_send, ny, MPIDataType, problem->rank + problem->npx * problem->npy - 1, BACK_WEST, back_west_receive, ny, MPIDataType, problem->rank + problem->npx * problem->npy - 1, FRONT_EAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, back_west_receive, 0, 1, dimz-1, ny, dimx, dimy, dimz);
        free(back_west_send);
        free(back_west_receive);
    }
    //exchange front corners
    if(problem->pz > 0){
        //exchange front_ne if got front_ne
        if(problem->py > 0 && problem->px < problem->npx - 1){
            DataType front_ne_send;
            CHECK_CUDA(cudaMemcpy(&front_ne_send, x_d + dimx * dimy + dimx + dimx - 2, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_ne_receive;
            MPI_Sendrecv(&front_ne_send, 1, MPIDataType, problem->rank - problem->npx * (problem->npy+1) + 1, FRONT_NE, &front_ne_receive, 1, MPIDataType, problem->rank - problem->npx * (problem->npy+1) + 1, BACK_SW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1, &front_ne_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
        //exchange front_se if got front_se
        if(problem->py < problem->npy - 1 && problem->px < problem->npx - 1){
            DataType front_se_send;
            CHECK_CUDA(cudaMemcpy(&front_se_send, x_d + dimx * dimy + dimx + dimx - 2 + (dimy-3)*dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_se_receive;
            MPI_Sendrecv(&front_se_send, 1, MPIDataType, problem->rank - problem->npx * (problem->npy-1) + 1, FRONT_SE, &front_se_receive, 1, MPIDataType, problem->rank - problem->npx * (problem->npy-1) + 1, BACK_NW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy - 1, &front_se_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
        //exchange front_sw if got front_sw
        if(problem->py < problem->npy - 1 && problem->px > 0){
            DataType front_sw_send;
            CHECK_CUDA(cudaMemcpy(&front_sw_send, x_d + dimx * dimy + dimx + 1 + (ny-1) * dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_sw_receive;
            MPI_Sendrecv(&front_sw_send, 1, MPIDataType, problem->rank - problem->npx * (problem->npy-1) - 1, FRONT_SW, &front_sw_receive, 1, MPIDataType, problem->rank - problem->npx * (problem->npy-1) - 1, BACK_NE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + (dimy -1) * dimx, &front_sw_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
        //exchange front_nw if got front_nw
        if(problem->py > 0 && problem->px > 0){
            DataType front_nw_send;
            CHECK_CUDA(cudaMemcpy(&front_nw_send, x_d + dimx * dimy + 1 + dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_nw_receive;
            MPI_Sendrecv(&front_nw_send, 1, MPIDataType, problem->rank - problem->npx * (problem->npy+1) - 1, FRONT_NW, &front_nw_receive, 1, MPIDataType, problem->rank - problem->npx * (problem->npy+1) - 1, BACK_SE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d, &front_nw_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }
    //exchange back corners
    if(problem->pz < problem->npz - 1){
        //exchange back_ne if got back_ne
        if(problem->py > 0 && problem->px < problem->npx - 1){
            DataType back_ne_send;
            CHECK_CUDA(cudaMemcpy(&back_ne_send, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_ne_receive;
            MPI_Sendrecv(&back_ne_send, 1, MPIDataType, problem->rank + problem->npx * (problem->npy-1) + 1, BACK_NE, &back_ne_receive, 1, MPIDataType, problem->rank + problem->npx * (problem->npy-1) + 1, FRONT_SW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + dimx * dimy * (dimz - 1), &back_ne_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
        //exchange back_se if got back_se
        if(problem->py < problem->npy - 1 && problem->px < problem->npx - 1){
            DataType back_se_send;
            CHECK_CUDA(cudaMemcpy(&back_se_send, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx + (dimy-3) * dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_se_receive;
            MPI_Sendrecv(&back_se_send, 1, MPIDataType, problem->rank + problem->npx * (problem->npy+1) + 1, BACK_SE, &back_se_receive, 1, MPIDataType, problem->rank + problem->npx * (problem->npy+1) + 1, FRONT_NW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + (dimy - 1) * dimx + dimx * dimy * (dimz - 1), &back_se_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
        //exchange back_sw if got back_sw
        if(problem->py < problem->npy - 1 && problem->px > 0){
            DataType back_sw_send;
            CHECK_CUDA(cudaMemcpy(&back_sw_send, x_d + dimx * dimy * (dimz - 2) + 1 + dimx + (ny-1) * dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_sw_receive;
            MPI_Sendrecv(&back_sw_send, 1, MPIDataType, problem->rank + problem->npx * (problem->npy+1) - 1, BACK_SW, &back_sw_receive, 1, MPIDataType, problem->rank + problem->npx * (problem->npy+1) - 1, FRONT_NE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx * (dimy - 1) + dimx * dimy * (dimz - 1), &back_sw_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
        //exchange back_nw if got back_nw
        if(problem->py > 0 && problem->px > 0){
            DataType back_nw_send;
            CHECK_CUDA(cudaMemcpy(&back_nw_send, x_d + dimx * dimy * (dimz - 2) + 1 + dimx, sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_nw_receive;
            MPI_Sendrecv(&back_nw_send, 1, MPIDataType, problem->rank + problem->npx * (problem->npy-1) - 1, BACK_NW, &back_nw_receive, 1, MPIDataType, problem->rank + problem->npx * (problem->npy-1) - 1, FRONT_SE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy * (dimz-1), &back_nw_receive, sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}


void extract_horizontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h += length_X;
        x_d += dimx * dimy;
    }
}

void inject_horizontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx * dimy;
        x_h += length_X;
    }
}

void extract_vertical_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        for(int j = 0; j < length_Y; j++){
            CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
            x_h++;
            x_d += dimx;
        }
        x_d += dimx * (dimy - length_Y);
    }
}

void inject_vertical_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        for(int j = 0; j < length_Y; j++){
            CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
            x_d += dimx;
            x_h ++;
        }
        x_d += dimx * (dimy - length_Y);
    }
}

void extract_frontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Y; i++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h += length_X;
        x_d += dimx;
    }
}

void inject_frontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Y; i++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx;
        x_h += length_X;
    }

}

void extract_edge_X_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
}

void inject_edge_X_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
}

void extract_edge_Y_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Y; j++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h++;
        x_d += dimx;
    }
}

void inject_edge_Y_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Y; j++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx;
        x_h ++;
    }
}

void extract_edge_Z_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Z; j++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h++;
        x_d += dimx * dimy;
    }
}

void inject_edge_Z_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Z; j++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx * dimy;
        x_h ++;
    }
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
    local_int_t data_paket = 0;
    for(int i = 0; i<problem->gnz; i++){ // go through all gnz layers
        int pz_recv = i / problem->nz;
        for(int j = 0; j<problem->gny; j++){ // go through all gny rows
            int py_recv = j / problem->ny;
            for(int l = 0; l<problem->npx; l++){ // go thorugh all nx columns
                int px_recv = l;
                int rank_recv = pz_recv * problem->npx * problem->npy + py_recv * problem->npx + px_recv;
                if(px_recv == problem->px && py_recv == problem->py && pz_recv == problem->pz){ //gathering rank holds the data
                    CHECK_CUDA(cudaMemcpy(result_h, own_data_d, problem->nx * sizeof(DataType), cudaMemcpyDeviceToHost));
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
    global_int_t gi0 = problem->gi0; //global index of local (0,0,0)
    
    local_int_t num_rows = nx * ny * nz;
    local_int_t num_cols = nx * ny * nz;
    
    for(int iz = 0; iz < nz; iz++){
        for(int iy = 0; iy < ny; iy++){
            for(int ix = 0; ix < nx; ix++){

                int i = ix + nx * iy + nx * ny * iz;

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