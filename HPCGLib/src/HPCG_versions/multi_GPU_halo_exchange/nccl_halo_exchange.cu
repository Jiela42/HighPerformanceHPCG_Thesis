#include "HPCG_versions/nccl_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

// Copies to host and back to device, no device-to-device copy
// TODO: Replace malloc per exchange with malloc once at beginning
template <typename T>
void nccl_Implementation<T>::ExchangeHaloNCCL(Halo *halo, Problem *problem) {

    int dimx = halo->dimx;
    int dimy = halo->dimy;
    int dimz = halo->dimz;
    int nx = halo->nx;
    int ny = halo->ny;
    int nz = halo->nz;
    DataType *x_d = halo->x_d;

    // We now need twice as many requests per exchange
    MPI_Request requests[52];
    int msg_count = 0;

    // Exchange north if got north
    if (problem->py > 0) {
        extract_horizontal_plane_from_GPU(x_d, halo->north_send_buff_h, 1, 1, 1, nx, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->north_recv_buff_h, nx * nz, MPIDataType,
                  problem->rank - problem->npx, SOUTH,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->north_send_buff_h, nx * nz, MPIDataType,
                  problem->rank - problem->npx, NORTH,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange east if got east
    if (problem->px < problem->npx - 1) {
        extract_vertical_plane_from_GPU(x_d, halo->east_send_buff_h, dimx - 2, 1, 1, ny, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->east_recv_buff_h, ny * nz, MPIDataType,
                  problem->rank + 1, WEST,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->east_send_buff_h, ny * nz, MPIDataType,
                  problem->rank + 1, EAST,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange south if got south
    if (problem->py < problem->npy - 1) {
        extract_horizontal_plane_from_GPU(x_d, halo->south_send_buff_h, 1, dimy - 2, 1, nx, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->south_recv_buff_h, nx * nz, MPIDataType,
                  problem->rank + problem->npx, NORTH,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->south_send_buff_h, nx * nz, MPIDataType,
                  problem->rank + problem->npx, SOUTH,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange west if got west
    if (problem->px > 0) {
        extract_vertical_plane_from_GPU(x_d, halo->west_send_buff_h, 1, 1, 1, ny, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->west_recv_buff_h, ny * nz, MPIDataType,
                  problem->rank - 1, EAST,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->west_send_buff_h, ny * nz, MPIDataType,
                  problem->rank - 1, WEST,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange ne if got ne
    if (problem->px < problem->npx - 1 && problem->py > 0) {
        extract_edge_Z_from_GPU(x_d, halo->ne_send_buff_h, dimx - 2, 1, 1, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->ne_recv_buff_h, nz, MPIDataType,
                  problem->rank - problem->npx + 1, SW,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->ne_send_buff_h, nz, MPIDataType,
                  problem->rank - problem->npx + 1, NE,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange se if got se
    if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
        extract_edge_Z_from_GPU(x_d, halo->se_send_buff_h, dimx - 2, dimy - 2, 1, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->se_recv_buff_h, nz, MPIDataType,
                  problem->rank + problem->npx + 1, NW,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->se_send_buff_h, nz, MPIDataType,
                  problem->rank + problem->npx + 1, SE,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange sw if got sw
    if (problem->px > 0 && problem->py < problem->npy - 1) {
        extract_edge_Z_from_GPU(x_d, halo->sw_send_buff_h, 1, dimy - 2, 1, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->sw_recv_buff_h, nz, MPIDataType,
                  problem->rank + problem->npx - 1, NE,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->sw_send_buff_h, nz, MPIDataType,
                  problem->rank + problem->npx - 1, SW,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange nw if got nw
    if (problem->py > 0 && problem->px > 0) {
        extract_edge_Z_from_GPU(x_d, halo->nw_send_buff_h, 1, 1, 1, nz, dimx, dimy, dimz);
        MPI_Irecv(halo->nw_recv_buff_h, nz, MPIDataType,
                  problem->rank - problem->npx - 1, SE,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->nw_send_buff_h, nz, MPIDataType,
                  problem->rank - problem->npx - 1, NW,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange front if got front
    if (problem->pz > 0) {
        extract_frontal_plane_from_GPU(x_d, halo->front_send_buff_h, 1, 1, 1, nx, ny, dimx, dimy, dimz);
        MPI_Irecv(halo->front_recv_buff_h, nx * ny, MPIDataType,
                  problem->rank - problem->npx * problem->npy, BACK,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->front_send_buff_h, nx * ny, MPIDataType,
                  problem->rank - problem->npx * problem->npy, FRONT,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange back if got back
    if (problem->pz < problem->npz - 1) {
        extract_frontal_plane_from_GPU(x_d, halo->back_send_buff_h, 1, 1, dimz - 2, nx, ny, dimx, dimy, dimz);
        MPI_Irecv(halo->back_recv_buff_h, nx * ny, MPIDataType,
                  problem->rank + problem->npx * problem->npy, FRONT,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->back_send_buff_h, nx * ny, MPIDataType,
                  problem->rank + problem->npx * problem->npy, BACK,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange front_north if got front_north
    if (problem->py > 0 && problem->pz > 0) {
        extract_edge_X_from_GPU(x_d, halo->front_north_send_buff_h, 1, 1, 1, nx, dimx, dimy, dimz);
        MPI_Irecv(halo->front_north_recv_buff_h, nx, MPIDataType,
                  problem->rank - problem->npx * (problem->npy + 1), BACK_SOUTH,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->front_north_send_buff_h, nx, MPIDataType,
                  problem->rank - problem->npx * (problem->npy + 1), FRONT_NORTH,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange front_east if got front_east
    if (problem->px < problem->npx - 1 && problem->pz > 0) {
        extract_edge_Y_from_GPU(x_d, halo->front_east_send_buff_h, dimx - 2, 1, 1, ny, dimx, dimy, dimz);
        MPI_Irecv(halo->front_east_recv_buff_h, ny, MPIDataType,
                  problem->rank - problem->npx * problem->npy + 1, BACK_WEST,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->front_east_send_buff_h, ny, MPIDataType,
                  problem->rank - problem->npx * problem->npy + 1, FRONT_EAST,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange front_south if got front_south
    if (problem->py < problem->npy - 1 && problem->pz > 0) {
        extract_edge_X_from_GPU(x_d, halo->front_south_send_buff_h, 1, dimy - 2, 1, nx, dimx, dimy, dimz);
        MPI_Irecv(halo->front_south_recv_buff_h, nx, MPIDataType,
                  problem->rank - problem->npx * (problem->npy - 1), BACK_NORTH,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->front_south_send_buff_h, nx, MPIDataType,
                  problem->rank - problem->npx * (problem->npy - 1), FRONT_SOUTH,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange front_west if got front_west
    if (problem->px > 0 && problem->pz > 0) {
        extract_edge_Y_from_GPU(x_d, halo->front_west_send_buff_h, 1, 1, 1, ny, dimx, dimy, dimz);
        MPI_Irecv(halo->front_west_recv_buff_h, ny, MPIDataType,
                  problem->rank - problem->npx * problem->npy - 1, BACK_EAST,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->front_west_send_buff_h, ny, MPIDataType,
                  problem->rank - problem->npx * problem->npy - 1, FRONT_WEST,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange back_north if got back_north
    if (problem->py > 0 && problem->pz < problem->npz - 1) {
        extract_edge_X_from_GPU(x_d, halo->back_north_send_buff_h, 1, 1, dimz - 2, nx, dimx, dimy, dimz);
        MPI_Irecv(halo->back_north_recv_buff_h, nx, MPIDataType,
                  problem->rank + problem->npx * (problem->npy - 1), FRONT_SOUTH,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->back_north_send_buff_h, nx, MPIDataType,
                  problem->rank + problem->npx * (problem->npy - 1), BACK_NORTH,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange back_east if got back_east
    if (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1) {
        extract_edge_Y_from_GPU(x_d, halo->back_east_send_buff_h, dimx - 2, 1, dimz - 2, ny, dimx, dimy, dimz);
        MPI_Irecv(halo->back_east_recv_buff_h, ny, MPIDataType,
                  problem->rank + problem->npx * problem->npy + 1, FRONT_WEST,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->back_east_send_buff_h, ny, MPIDataType,
                  problem->rank + problem->npx * problem->npy + 1, BACK_EAST,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange back_south if got back_south
    if (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1) {
        extract_edge_X_from_GPU(x_d, halo->back_south_send_buff_h, 1, dimy - 2, dimz - 2, nx, dimx, dimy, dimz);
        MPI_Irecv(halo->back_south_recv_buff_h, nx, MPIDataType,
                  problem->rank + problem->npx * (problem->npy + 1), FRONT_NORTH,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->back_south_send_buff_h, nx, MPIDataType,
                  problem->rank + problem->npx * (problem->npy + 1), BACK_SOUTH,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange back_west if got back_west
    if (problem->px > 0 && problem->pz < problem->npz - 1) {
        extract_edge_Y_from_GPU(x_d, halo->back_west_send_buff_h, 1, 1, dimz - 2, ny, dimx, dimy, dimz);
        MPI_Irecv(halo->back_west_recv_buff_h, ny, MPIDataType,
                  problem->rank + problem->npx * problem->npy - 1, FRONT_EAST,
                  MPI_COMM_WORLD, &requests[msg_count]);
        MPI_Isend(halo->back_west_send_buff_h, ny, MPIDataType,
                  problem->rank + problem->npx * problem->npy - 1, BACK_WEST,
                  MPI_COMM_WORLD, &requests[msg_count + 1]);
        msg_count += 2;
    }

    // Exchange front corners
    if (problem->pz > 0) {
        // Exchange front_ne if got front_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->front_ne_send_buff_h, x_d + dimx * dimy + dimx + dimx - 2,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->front_ne_recv_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy + 1) + 1, BACK_SW,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->front_ne_send_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy + 1) + 1, FRONT_NE,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
        
        // Exchange front_se if got front_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->front_se_send_buff_h, x_d + dimx * dimy + dimx + dimx - 2 + (dimy - 3) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->front_se_recv_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy - 1) + 1, BACK_NW,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->front_se_send_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy - 1) + 1, FRONT_SE,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
        // Exchange front_sw if got front_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->front_sw_send_buff_h, x_d + dimx * dimy + dimx + 1 + (ny - 1) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->front_sw_recv_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy - 1) - 1, BACK_NE,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->front_sw_send_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy - 1) - 1, FRONT_SW,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
        // Exchange front_nw if got front_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->front_nw_send_buff_h, x_d + dimx * dimy + 1 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->front_nw_recv_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy + 1) - 1, BACK_SE,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->front_nw_send_buff_h, 1, MPIDataType,
                      problem->rank - problem->npx * (problem->npy + 1) - 1, FRONT_NW,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
    }

    // Exchange back corners
    if (problem->pz < problem->npz - 1) {
        // Exchange back_ne if got back_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->back_ne_send_buff_h, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->back_ne_recv_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy - 1) + 1, FRONT_SW,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->back_ne_send_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy - 1) + 1, BACK_NE,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
        // Exchange back_se if got back_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->back_se_send_buff_h, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx + (dimy - 3) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->back_se_recv_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy + 1) + 1, FRONT_NW,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->back_se_send_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy + 1) + 1, BACK_SE,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
        // Exchange back_sw if got back_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->back_sw_send_buff_h, x_d + dimx * dimy * (dimz - 2) + 1 + dimx + (ny - 1) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->back_sw_recv_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy + 1) - 1, FRONT_NE,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->back_sw_send_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy + 1) - 1, BACK_SW,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
        // Exchange back_nw if got back_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->back_nw_send_buff_h, x_d + dimx * dimy * (dimz - 2) + 1 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Irecv(halo->back_nw_recv_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy - 1) - 1, FRONT_SE,
                      MPI_COMM_WORLD, &requests[msg_count]);
            MPI_Isend(halo->back_nw_send_buff_h, 1, MPIDataType,
                      problem->rank + problem->npx * (problem->npy - 1) - 1, BACK_NW,
                      MPI_COMM_WORLD, &requests[msg_count + 1]);
            msg_count += 2;
        }
    }
    
    // Wait until all exchanges are done
    MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE);

    // Now that we received all data, we can inject it back to the halo

    // Inject north if got north
    if (problem->py > 0) {
        inject_horizontal_plane_to_GPU(x_d, halo->north_recv_buff_h, 1, 0, 1, nx, nz, dimx, dimy, dimz);
    }

    // Inject east if got east
    if (problem->px < problem->npx - 1) {
        inject_vertical_plane_to_GPU(x_d, halo->east_recv_buff_h, dimx - 1, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Inject south if got south
    if (problem->py < problem->npy - 1) {
        inject_horizontal_plane_to_GPU(x_d, halo->south_recv_buff_h, 1, dimy - 1, 1, nx, nz, dimx, dimy, dimz);
    }

    // Inject west if got west
    if (problem->px > 0) {
        inject_vertical_plane_to_GPU(x_d, halo->west_recv_buff_h, 0, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Inject ne if got ne
    if (problem->px < problem->npx - 1 && problem->py > 0) {
        inject_edge_Z_to_GPU(x_d, halo->ne_recv_buff_h, dimx - 1, 0, 1, nz, dimx, dimy, dimz);
    }

    // Inject se if got se
    if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
        inject_edge_Z_to_GPU(x_d, halo->se_recv_buff_h, dimx - 1, dimy - 1, 1, nz, dimx, dimy, dimz);
    }

    // Inject sw if got sw
    if (problem->px > 0 && problem->py < problem->npy - 1) {
        inject_edge_Z_to_GPU(x_d, halo->sw_recv_buff_h, 0, dimy - 1, 1, nz, dimx, dimy, dimz);
    }

    // Inject nw if got nw
    if (problem->py > 0 && problem->px > 0) {
        inject_edge_Z_to_GPU(x_d, halo->nw_recv_buff_h, 0, 0, 1, nz, dimx, dimy, dimz);
    }

    // Inject front if got front
    if (problem->pz > 0) {
        inject_frontal_plane_to_GPU(x_d, halo->front_recv_buff_h, 1, 1, 0, nx, ny, dimx, dimy, dimz);
    }

    // Inject back if got back
    if (problem->pz < problem->npz - 1) {
        inject_frontal_plane_to_GPU(x_d, halo->back_recv_buff_h, 1, 1, dimz - 1, nx, ny, dimx, dimy, dimz);
    }

    // Inject front_north if got front_north
    if (problem->py > 0 && problem->pz > 0) {
        inject_edge_X_to_GPU(x_d, halo->front_north_recv_buff_h, 1, 0, 0, nx, dimx, dimy, dimz);
    }

    // Inject front_east if got front_east
    if (problem->px < problem->npx - 1 && problem->pz > 0) {
        inject_edge_Y_to_GPU(x_d, halo->front_east_recv_buff_h, dimx - 1, 1, 0, ny, dimx, dimy, dimz);
    }

    // Inject front_south if got front_south
    if (problem->py < problem->npy - 1 && problem->pz > 0) {
        inject_edge_X_to_GPU(x_d, halo->front_south_recv_buff_h, 1, dimy - 1, 0, nx, dimx, dimy, dimz);
    }

    // Inject front_west if got front_west
    if (problem->px > 0 && problem->pz > 0) {
        inject_edge_Y_to_GPU(x_d, halo->front_west_recv_buff_h, 0, 1, 0, ny, dimx, dimy, dimz);
    }

    // Inject back_north if got back_north
    if (problem->py > 0 && problem->pz < problem->npz - 1) {
        inject_edge_X_to_GPU(x_d, halo->back_north_recv_buff_h, 1, 0, dimz - 1, nx, dimx, dimy, dimz);
    }

    // Inject back_east if got back_east
    if (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1) {
        inject_edge_Y_to_GPU(x_d, halo->back_east_recv_buff_h, dimx - 1, 1, dimz - 1, ny, dimx, dimy, dimz);
    }

    // Inject back_south if got back_south
    if (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1) {
        inject_edge_X_to_GPU(x_d, halo->back_south_recv_buff_h, 1, dimy - 1, dimz - 1, nx, dimx, dimy, dimz);
    }

    // Inject back_west if got back_west
    if (problem->px > 0 && problem->pz < problem->npz - 1) {
        inject_edge_Y_to_GPU(x_d, halo->back_west_recv_buff_h, 0, 1, dimz - 1, ny, dimx, dimy, dimz);
    }

    // Inject front corners
    if (problem->pz > 0) {
        // Inject front_ne if got front_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1, halo->front_ne_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject front_se if got front_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy - 1, halo->front_se_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject front_sw if got front_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d + (dimy - 1) * dimx, halo->front_sw_recv_buff_h, 
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject front_nw if got front_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d, halo->front_nw_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }

    // Inject back corners
    if (problem->pz < problem->npz - 1) {
        // Inject back_ne if got back_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + dimx * dimy * (dimz - 1), halo->back_ne_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject back_se if got back_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + (dimy - 1) * dimx + dimx * dimy * (dimz - 1), halo->back_se_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject back_sw if got back_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx * (dimy - 1) + dimx * dimy * (dimz - 1), halo->back_sw_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject back_nw if got back_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy * (dimz - 1), halo->back_nw_recv_buff_h,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }

}

// Explicit template instantiation
template class NCCL_Implementation<DataType>;