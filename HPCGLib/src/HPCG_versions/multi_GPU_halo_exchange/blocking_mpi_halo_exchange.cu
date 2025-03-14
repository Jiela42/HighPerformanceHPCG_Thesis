#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

//copies to host and back to device, no device-to-device copy
//TODO: Replace malloc per exchange with malloc once at beginning
template <typename T>
void blocking_mpi_Implementation<T>::ExchangeHaloBlockingMPI(Halo *halo, Problem *problem){
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


// explicit template instantiation
template class blocking_mpi_Implementation <DataType>;