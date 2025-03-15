#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

// Copies to host and back to device, no device-to-device copy
// TODO: Replace malloc per exchange with malloc once at beginning
template <typename T>
void blocking_mpi_Implementation<T>::ExchangeHaloBlockingMPI(Halo *halo, Problem *problem) {

    int dimx = halo->dimx;
    int dimy = halo->dimy;
    int dimz = halo->dimz;
    int nx = halo->nx;
    int ny = halo->ny;
    int nz = halo->nz;
    DataType *x_d = halo->x_d;

    // Exchange north if got north
    if (problem->py > 0) {
        extract_horizontal_plane_from_GPU(x_d, halo->north_send_buff_h, 1, 1, 1, nx, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->north_send_buff_h, nx * nz, MPIDataType,
                     problem->rank - problem->npx, NORTH,
                     halo->north_recv_buff_h, nx * nz, MPIDataType,
                     problem->rank - problem->npx, SOUTH,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_horizontal_plane_to_GPU(x_d, halo->north_recv_buff_h, 1, 0, 1, nx, nz, dimx, dimy, dimz);
    }

    // Exchange east if got east
    if (problem->px < problem->npx - 1) {
        extract_vertical_plane_from_GPU(x_d, halo->east_send_buff_h, dimx - 2, 1, 1, ny, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->east_send_buff_h, ny * nz, MPIDataType,
                     problem->rank + 1, EAST,
                     halo->east_recv_buff_h, ny * nz, MPIDataType,
                     problem->rank + 1, WEST,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_vertical_plane_to_GPU(x_d, halo->east_recv_buff_h, dimx - 1, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Exchange south if got south
    if (problem->py < problem->npy - 1) {
        extract_horizontal_plane_from_GPU(x_d, halo->south_send_buff_h, 1, dimy - 2, 1, nx, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->south_send_buff_h, nx * nz, MPIDataType,
                     problem->rank + problem->npx, SOUTH,
                     halo->south_recv_buff_h, nx * nz, MPIDataType,
                     problem->rank + problem->npx, NORTH,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_horizontal_plane_to_GPU(x_d, halo->south_recv_buff_h, 1, dimy - 1, 1, nx, nz, dimx, dimy, dimz);
    }

    // Exchange west if got west
    if (problem->px > 0) {
        extract_vertical_plane_from_GPU(x_d, halo->west_send_buff_h, 1, 1, 1, ny, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->west_send_buff_h, ny * nz, MPIDataType,
                     problem->rank - 1, WEST,
                     halo->west_recv_buff_h, ny * nz, MPIDataType,
                     problem->rank - 1, EAST,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_vertical_plane_to_GPU(x_d, halo->west_recv_buff_h, 0, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Exchange ne if got ne
    if (problem->px < problem->npx - 1 && problem->py > 0) {
        extract_edge_Z_from_GPU(x_d, halo->ne_send_buff_h, dimx - 2, 1, 1, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->ne_send_buff_h, nz, MPIDataType,
                     problem->rank - problem->npx + 1, NE,
                     halo->ne_recv_buff_h, nz, MPIDataType,
                     problem->rank - problem->npx + 1, SW,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, halo->ne_recv_buff_h, dimx - 1, 0, 1, nz, dimx, dimy, dimz);
    }

    // Exchange se if got se
    if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
        extract_edge_Z_from_GPU(x_d, halo->se_send_buff_h, dimx - 2, dimy - 2, 1, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->se_send_buff_h, nz, MPIDataType,
                     problem->rank + problem->npx + 1, SE,
                     halo->se_recv_buff_h, nz, MPIDataType,
                     problem->rank + problem->npx + 1, NW,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, halo->se_recv_buff_h, dimx - 1, dimy - 1, 1, nz, dimx, dimy, dimz);
    }

    // Exchange sw if got sw
    if (problem->px > 0 && problem->py < problem->npy - 1) {
        extract_edge_Z_from_GPU(x_d, halo->sw_send_buff_h, 1, dimy - 2, 1, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->sw_send_buff_h, nz, MPIDataType,
                     problem->rank + problem->npx - 1, SW,
                     halo->sw_recv_buff_h, nz, MPIDataType,
                     problem->rank + problem->npx - 1, NE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, halo->sw_recv_buff_h, 0, dimy - 1, 1, nz, dimx, dimy, dimz);
    }

    // Exchange nw if got nw
    if (problem->py > 0 && problem->px > 0) {
        extract_edge_Z_from_GPU(x_d, halo->nw_send_buff_h, 1, 1, 1, nz, dimx, dimy, dimz);
        MPI_Sendrecv(halo->nw_send_buff_h, nz, MPIDataType,
                     problem->rank - problem->npx - 1, NW,
                     halo->nw_recv_buff_h, nz, MPIDataType,
                     problem->rank - problem->npx - 1, SE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Z_to_GPU(x_d, halo->nw_recv_buff_h, 0, 0, 1, nz, dimx, dimy, dimz);
    }

    // Exchange front if got front
    if (problem->pz > 0) {
        extract_frontal_plane_from_GPU(x_d, halo->front_send_buff_h, 1, 1, 1, nx, ny, dimx, dimy, dimz);
        MPI_Sendrecv(halo->front_send_buff_h, nx * ny, MPIDataType,
                     problem->rank - problem->npx * problem->npy, FRONT,
                     halo->front_recv_buff_h, nx * ny, MPIDataType,
                     problem->rank - problem->npx * problem->npy, BACK,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_frontal_plane_to_GPU(x_d, halo->front_recv_buff_h, 1, 1, 0, nx, ny, dimx, dimy, dimz);
    }

    // Exchange back if got back
    if (problem->pz < problem->npz - 1) {
        extract_frontal_plane_from_GPU(x_d, halo->back_send_buff_h, 1, 1, dimz - 2, nx, ny, dimx, dimy, dimz);
        MPI_Sendrecv(halo->back_send_buff_h, nx * ny, MPIDataType,
                     problem->rank + problem->npx * problem->npy, BACK,
                     halo->back_recv_buff_h, nx * ny, MPIDataType,
                     problem->rank + problem->npx * problem->npy, FRONT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_frontal_plane_to_GPU(x_d, halo->back_recv_buff_h, 1, 1, dimz - 1, nx, ny, dimx, dimy, dimz);
    }

    // Exchange front_north if got front_north
    if (problem->py > 0 && problem->pz > 0) {
        extract_edge_X_from_GPU(x_d, halo->front_north_send_buff_h, 1, 1, 1, nx, dimx, dimy, dimz);
        MPI_Sendrecv(halo->front_north_send_buff_h, nx, MPIDataType,
                     problem->rank - problem->npx * (problem->npy + 1), FRONT_NORTH,
                     halo->front_north_recv_buff_h, nx, MPIDataType,
                     problem->rank - problem->npx * (problem->npy + 1), BACK_SOUTH,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, halo->front_north_recv_buff_h, 1, 0, 0, nx, dimx, dimy, dimz);
    }

    // Exchange front_east if got front_east
    if (problem->px < problem->npx - 1 && problem->pz > 0) {
        extract_edge_Y_from_GPU(x_d, halo->front_east_send_buff_h, dimx - 2, 1, 1, ny, dimx, dimy, dimz);
        MPI_Sendrecv(halo->front_east_send_buff_h, ny, MPIDataType,
                     problem->rank - problem->npx * problem->npy + 1, FRONT_EAST,
                     halo->front_east_recv_buff_h, ny, MPIDataType,
                     problem->rank - problem->npx * problem->npy + 1, BACK_WEST,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, halo->front_east_recv_buff_h, dimx - 1, 1, 0, ny, dimx, dimy, dimz);
    }

    // Exchange front_south if got front_south
    if (problem->py < problem->npy - 1 && problem->pz > 0) {
        extract_edge_X_from_GPU(x_d, halo->front_south_send_buff_h, 1, dimy - 2, 1, nx, dimx, dimy, dimz);
        MPI_Sendrecv(halo->front_south_send_buff_h, nx, MPIDataType,
                     problem->rank - problem->npx * (problem->npy - 1), FRONT_SOUTH,
                     halo->front_south_recv_buff_h, nx, MPIDataType,
                     problem->rank - problem->npx * (problem->npy - 1), BACK_NORTH,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, halo->front_south_recv_buff_h, 1, dimy - 1, 0, nx, dimx, dimy, dimz);
    }

    // Exchange front_west if got front_west
    if (problem->px > 0 && problem->pz > 0) {
        extract_edge_Y_from_GPU(x_d, halo->front_west_send_buff_h, 1, 1, 1, ny, dimx, dimy, dimz);
        MPI_Sendrecv(halo->front_west_send_buff_h, ny, MPIDataType,
                     problem->rank - problem->npx * problem->npy - 1, FRONT_WEST,
                     halo->front_west_recv_buff_h, ny, MPIDataType,
                     problem->rank - problem->npx * problem->npy - 1, BACK_EAST,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, halo->front_west_recv_buff_h, 0, 1, 0, ny, dimx, dimy, dimz);
    }

    // Exchange back_north if got back_north
    if (problem->py > 0 && problem->pz < problem->npz - 1) {
        extract_edge_X_from_GPU(x_d, halo->back_north_send_buff_h, 1, 1, dimz - 2, nx, dimx, dimy, dimz);
        MPI_Sendrecv(halo->back_north_send_buff_h, nx, MPIDataType,
                     problem->rank + problem->npx * (problem->npy - 1), BACK_NORTH,
                     halo->back_north_recv_buff_h, nx, MPIDataType,
                     problem->rank + problem->npx * (problem->npy - 1), FRONT_SOUTH,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, halo->back_north_recv_buff_h, 1, 0, dimz - 1, nx, dimx, dimy, dimz);
    }

    // Exchange back_east if got back_east
    if (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1) {
        extract_edge_Y_from_GPU(x_d, halo->back_east_send_buff_h, dimx - 2, 1, dimz - 2, ny, dimx, dimy, dimz);
        MPI_Sendrecv(halo->back_east_send_buff_h, ny, MPIDataType,
                     problem->rank + problem->npx * problem->npy + 1, BACK_EAST,
                     halo->back_east_recv_buff_h, ny, MPIDataType,
                     problem->rank + problem->npx * problem->npy + 1, FRONT_WEST,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, halo->back_east_recv_buff_h, dimx - 1, 1, dimz - 1, ny, dimx, dimy, dimz);
    }

    // Exchange back_south if got back_south
    if (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1) {
        extract_edge_X_from_GPU(x_d, halo->back_south_send_buff_h, 1, dimy - 2, dimz - 2, nx, dimx, dimy, dimz);
        MPI_Sendrecv(halo->back_south_send_buff_h, nx, MPIDataType,
                     problem->rank + problem->npx * (problem->npy + 1), BACK_SOUTH,
                     halo->back_south_recv_buff_h, nx, MPIDataType,
                     problem->rank + problem->npx * (problem->npy + 1), FRONT_NORTH,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_X_to_GPU(x_d, halo->back_south_recv_buff_h, 1, dimy - 1, dimz - 1, nx, dimx, dimy, dimz);
    }

    // Exchange back_west if got back_west
    if (problem->px > 0 && problem->pz < problem->npz - 1) {
        extract_edge_Y_from_GPU(x_d, halo->back_west_send_buff_h, 1, 1, dimz - 2, ny, dimx, dimy, dimz);
        MPI_Sendrecv(halo->back_west_send_buff_h, ny, MPIDataType,
                     problem->rank + problem->npx * problem->npy - 1, BACK_WEST,
                     halo->back_west_recv_buff_h, ny, MPIDataType,
                     problem->rank + problem->npx * problem->npy - 1, FRONT_EAST,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        inject_edge_Y_to_GPU(x_d, halo->back_west_recv_buff_h, 0, 1, dimz - 1, ny, dimx, dimy, dimz);
    }

    // Exchange front corners
    if (problem->pz > 0) {
        // Exchange front_ne if got front_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            DataType front_ne_send;
            CHECK_CUDA(cudaMemcpy(&front_ne_send, x_d + dimx * dimy + dimx + dimx - 2,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_ne_receive;
            MPI_Sendrecv(&front_ne_send, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy + 1) + 1, FRONT_NE,
                         &front_ne_receive, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy + 1) + 1, BACK_SW,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1, &front_ne_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Exchange front_se if got front_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            DataType front_se_send;
            CHECK_CUDA(cudaMemcpy(&front_se_send, x_d + dimx * dimy + dimx + dimx - 2 + (dimy - 3) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_se_receive;
            MPI_Sendrecv(&front_se_send, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy - 1) + 1, FRONT_SE,
                         &front_se_receive, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy - 1) + 1, BACK_NW,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy - 1, &front_se_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Exchange front_sw if got front_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            DataType front_sw_send;
            CHECK_CUDA(cudaMemcpy(&front_sw_send, x_d + dimx * dimy + dimx + 1 + (ny - 1) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_sw_receive;
            MPI_Sendrecv(&front_sw_send, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy - 1) - 1, FRONT_SW,
                         &front_sw_receive, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy - 1) - 1, BACK_NE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + (dimy - 1) * dimx, &front_sw_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Exchange front_nw if got front_nw
        if (problem->py > 0 && problem->px > 0) {
            DataType front_nw_send;
            CHECK_CUDA(cudaMemcpy(&front_nw_send, x_d + dimx * dimy + 1 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType front_nw_receive;
            MPI_Sendrecv(&front_nw_send, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy + 1) - 1, FRONT_NW,
                         &front_nw_receive, 1, MPIDataType,
                         problem->rank - problem->npx * (problem->npy + 1) - 1, BACK_SE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d, &front_nw_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }

    // Exchange back corners
    if (problem->pz < problem->npz - 1) {
        // Exchange back_ne if got back_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            DataType back_ne_send;
            CHECK_CUDA(cudaMemcpy(&back_ne_send, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_ne_receive;
            MPI_Sendrecv(&back_ne_send, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy - 1) + 1, BACK_NE,
                         &back_ne_receive, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy - 1) + 1, FRONT_SW,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + dimx * dimy * (dimz - 1), &back_ne_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Exchange back_se if got back_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            DataType back_se_send;
            CHECK_CUDA(cudaMemcpy(&back_se_send, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx + (dimy - 3) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_se_receive;
            MPI_Sendrecv(&back_se_send, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy + 1) + 1, BACK_SE,
                         &back_se_receive, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy + 1) + 1, FRONT_NW,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + (dimy - 1) * dimx + dimx * dimy * (dimz - 1), &back_se_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Exchange back_sw if got back_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            DataType back_sw_send;
            CHECK_CUDA(cudaMemcpy(&back_sw_send, x_d + dimx * dimy * (dimz - 2) + 1 + dimx + (ny - 1) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_sw_receive;
            MPI_Sendrecv(&back_sw_send, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy + 1) - 1, BACK_SW,
                         &back_sw_receive, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy + 1) - 1, FRONT_NE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx * (dimy - 1) + dimx * dimy * (dimz - 1), &back_sw_receive,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Exchange back_nw if got back_nw
        if (problem->py > 0 && problem->px > 0) {
            DataType back_nw_send;
            CHECK_CUDA(cudaMemcpy(&back_nw_send, x_d + dimx * dimy * (dimz - 2) + 1 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            DataType back_nw_receive;
            MPI_Sendrecv(&back_nw_send, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy - 1) - 1, BACK_NW,
                         &back_nw_receive, 1, MPIDataType,
                         problem->rank + problem->npx * (problem->npy - 1) - 1, FRONT_SE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy * (dimz - 1), &back_nw_receive,
    sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }

}

// Explicit template instantiation
template class blocking_mpi_Implementation<DataType>;