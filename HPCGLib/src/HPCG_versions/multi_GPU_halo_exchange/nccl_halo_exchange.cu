#include "HPCG_versions/nccl_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

#include "nccl.h"

//each process is using one GPU
#define NCCL_TYPE ncclDouble

//cf. https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html

static uint64_t getHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
      result = ((result << 5) + result) ^ string[c];
    }
    return result;
  }
  
/* Generate a hash of the unique identifying string for this host
* that will be unique for both bare-metal and container instances
* Equivalent of a hash of;
*
* $(hostname)$(cat /proc/sys/kernel/random/boot_id)
*
*/
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
    char hostHash[1024];

    // Fall back is the hostname if something fails
    (void) strncpy(hostHash, hostname, sizeof(hostHash));
    int offset = strlen(hostHash);

    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
        char *p;
        if (fscanf(file, "%ms", &p) == 1) {
            strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
            free(p);
        }
    }
    fclose(file);

    // Make sure the string is terminated
    hostHash[sizeof(hostHash)-1]='\0';

    return getHash(hostHash);
}
  
static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

template <typename T>
Problem* NCCL_Implementation<T>::init_comm_NCCL(int argc, char *argv[], int npx, int npy, int npz, int nx, int ny, int nz) {
    
    //initializing MPI
    CHECK_MPI(MPI_Init( &argc , &argv ));
    int size, rank, localRank = 0;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    //calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<size; p++) {
        if (p == rank) break;
        if (hostHashs[p] == hostHashs[rank]) localRank++;
    }

    //picking a GPU based on localRank
    CHECK_CUDA(cudaSetDevice(localRank));
    CHECK_CUDA(cudaStreamCreate(&this->cuda_stream));

    //get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    CHECK_MPI(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL
    CHECK_NCCL(ncclCommInitRank(&this->nccl_comm, size, id, rank));

    //initializing the problem
    Problem *problem = (Problem *)malloc(sizeof(Problem));
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, problem);

    return problem;
}

void extractData(Halo *halo, Problem *problem){
    int dimx = halo->dimx;
    int dimy = halo->dimy;
    int dimz = halo->dimz;
    int nx = halo->nx;
    int ny = halo->ny;
    int nz = halo->nz;
    DataType *x_d = halo->x_d;

    // Exchange north if got north
    if (problem->py > 0) {
        extract_horizontal_plane_from_GPU(x_d, halo->north_send_buff_d, 1, 1, 1, nx, nz, dimx, dimy, dimz);
    }

    // Exchange east if got east
    if (problem->px < problem->npx - 1) {
        extract_vertical_plane_from_GPU(x_d, halo->east_send_buff_d, dimx - 2, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Exchange south if got south
    if (problem->py < problem->npy - 1) {
        extract_horizontal_plane_from_GPU(x_d, halo->south_send_buff_d, 1, dimy - 2, 1, nx, nz, dimx, dimy, dimz);
    }

    // Exchange west if got west
    if (problem->px > 0) {
        extract_vertical_plane_from_GPU(x_d, halo->west_send_buff_d, 1, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Exchange ne if got ne
    if (problem->px < problem->npx - 1 && problem->py > 0) {
        extract_edge_Z_from_GPU(x_d, halo->ne_send_buff_d, dimx - 2, 1, 1, nz, dimx, dimy, dimz);
    }

    // Exchange se if got se
    if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
        extract_edge_Z_from_GPU(x_d, halo->se_send_buff_d, dimx - 2, dimy - 2, 1, nz, dimx, dimy, dimz);
    }

    // Exchange sw if got sw
    if (problem->px > 0 && problem->py < problem->npy - 1) {
        extract_edge_Z_from_GPU(x_d, halo->sw_send_buff_d, 1, dimy - 2, 1, nz, dimx, dimy, dimz);
    }

    // Exchange nw if got nw
    if (problem->py > 0 && problem->px > 0) {
        extract_edge_Z_from_GPU(x_d, halo->nw_send_buff_d, 1, 1, 1, nz, dimx, dimy, dimz);
    }

    // Exchange front if got front
    if (problem->pz > 0) {
        extract_frontal_plane_from_GPU(x_d, halo->front_send_buff_d, 1, 1, 1, nx, ny, dimx, dimy, dimz);
    }

    // Exchange back if got back
    if (problem->pz < problem->npz - 1) {
        extract_frontal_plane_from_GPU(x_d, halo->back_send_buff_d, 1, 1, dimz - 2, nx, ny, dimx, dimy, dimz);
    }

    // Exchange front_north if got front_north
    if (problem->py > 0 && problem->pz > 0) {
        extract_edge_X_from_GPU(x_d, halo->front_north_send_buff_d, 1, 1, 1, nx, dimx, dimy, dimz);
    }

    // Exchange front_east if got front_east
    if (problem->px < problem->npx - 1 && problem->pz > 0) {
        extract_edge_Y_from_GPU(x_d, halo->front_east_send_buff_d, dimx - 2, 1, 1, ny, dimx, dimy, dimz);
    }

    // Exchange front_south if got front_south
    if (problem->py < problem->npy - 1 && problem->pz > 0) {
        extract_edge_X_from_GPU(x_d, halo->front_south_send_buff_d, 1, dimy - 2, 1, nx, dimx, dimy, dimz);
    }

    // Exchange front_west if got front_west
    if (problem->px > 0 && problem->pz > 0) {
        extract_edge_Y_from_GPU(x_d, halo->front_west_send_buff_d, 1, 1, 1, ny, dimx, dimy, dimz);
    }

    // Exchange back_north if got back_north
    if (problem->py > 0 && problem->pz < problem->npz - 1) {
        extract_edge_X_from_GPU(x_d, halo->back_north_send_buff_d, 1, 1, dimz - 2, nx, dimx, dimy, dimz);
    }

    // Exchange back_east if got back_east
    if (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1) {
        extract_edge_Y_from_GPU(x_d, halo->back_east_send_buff_d, dimx - 2, 1, dimz - 2, ny, dimx, dimy, dimz);

    }

    // Exchange back_south if got back_south
    if (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1) {
        extract_edge_X_from_GPU(x_d, halo->back_south_send_buff_d, 1, dimy - 2, dimz - 2, nx, dimx, dimy, dimz);
    }

    // Exchange back_west if got back_west
    if (problem->px > 0 && problem->pz < problem->npz - 1) {
        extract_edge_Y_from_GPU(x_d, halo->back_west_send_buff_d, 1, 1, dimz - 2, ny, dimx, dimy, dimz);
    }

    // Exchange front corners
    if (problem->pz > 0) {
        // Exchange front_ne if got front_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->front_ne_send_buff_d, x_d + dimx * dimy + dimx + dimx - 2,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
        
        // Exchange front_se if got front_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->front_se_send_buff_d, x_d + dimx * dimy + dimx + dimx - 2 + (dimy - 3) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
        // Exchange front_sw if got front_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->front_sw_send_buff_d, x_d + dimx * dimy + dimx + 1 + (ny - 1) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
        // Exchange front_nw if got front_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->front_nw_send_buff_d, x_d + dimx * dimy + 1 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
    }

    // Exchange back corners
    if (problem->pz < problem->npz - 1) {
        // Exchange back_ne if got back_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->back_ne_send_buff_d, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
        // Exchange back_se if got back_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(halo->back_se_send_buff_d, x_d + dimx * dimy * (dimz - 2) + dimx - 2 + dimx + (dimy - 3) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
        // Exchange back_sw if got back_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->back_sw_send_buff_d, x_d + dimx * dimy * (dimz - 2) + 1 + dimx + (ny - 1) * dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
        // Exchange back_nw if got back_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(halo->back_nw_send_buff_d, x_d + dimx * dimy * (dimz - 2) + 1 + dimx,
                                    sizeof(DataType), cudaMemcpyDeviceToHost));
        }
    }
}

void sendRecvData(Halo *halo, Problem *problem, ncclComm_t& nccl_comm, cudaStream_t& cuda_stream){
    int dimx = halo->dimx;
    int dimy = halo->dimy;
    int dimz = halo->dimz;
    int nx = halo->nx;
    int ny = halo->ny;
    int nz = halo->nz;
    DataType *x_d = halo->x_d;

    ncclGroupStart();

    // Exchange north if available
    if (problem->py > 0) {
        CHECK_NCCL(ncclRecv(halo->north_recv_buff_d, nx * nz, NCCL_TYPE,
                            problem->rank - problem->npx, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->north_send_buff_d, nx * nz, NCCL_TYPE,
                            problem->rank - problem->npx, nccl_comm, cuda_stream));
    }

    // Exchange east if available
    if (problem->px < problem->npx - 1) {
        CHECK_NCCL(ncclRecv(halo->east_recv_buff_d, ny * nz, NCCL_TYPE,
                            problem->rank + 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->east_send_buff_d, ny * nz, NCCL_TYPE,
                            problem->rank + 1, nccl_comm, cuda_stream));
    }

    // Exchange south if available
    if (problem->py < problem->npy - 1) {
        CHECK_NCCL(ncclRecv(halo->south_recv_buff_d, nx * nz, NCCL_TYPE,
                            problem->rank + problem->npx, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->south_send_buff_d, nx * nz, NCCL_TYPE,
                            problem->rank + problem->npx, nccl_comm, cuda_stream));
    }

    // Exchange west if available
    if (problem->px > 0) {
        CHECK_NCCL(ncclRecv(halo->west_recv_buff_d, ny * nz, NCCL_TYPE,
                            problem->rank - 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->west_send_buff_d, ny * nz, NCCL_TYPE,
                            problem->rank - 1, nccl_comm, cuda_stream));
    }

    // Exchange northeast if available
    if (problem->px < problem->npx - 1 && problem->py > 0) {
        CHECK_NCCL(ncclRecv(halo->ne_recv_buff_d, nz, NCCL_TYPE,
                            problem->rank - problem->npx + 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->ne_send_buff_d, nz, NCCL_TYPE,
                            problem->rank - problem->npx + 1, nccl_comm, cuda_stream));
    }

    // Exchange southeast if available
    if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
        CHECK_NCCL(ncclRecv(halo->se_recv_buff_d, nz, NCCL_TYPE,
                            problem->rank + problem->npx + 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->se_send_buff_d, nz, NCCL_TYPE,
                            problem->rank + problem->npx + 1, nccl_comm, cuda_stream));
    }

    // Exchange southwest if available
    if (problem->px > 0 && problem->py < problem->npy - 1) {
        CHECK_NCCL(ncclRecv(halo->sw_recv_buff_d, nz, NCCL_TYPE,
                            problem->rank + problem->npx - 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->sw_send_buff_d, nz, NCCL_TYPE,
                            problem->rank + problem->npx - 1, nccl_comm, cuda_stream));
    }

    // Exchange northwest if available
    if (problem->py > 0 && problem->px > 0) {
        CHECK_NCCL(ncclRecv(halo->nw_recv_buff_d, nz, NCCL_TYPE,
                            problem->rank - problem->npx - 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->nw_send_buff_d, nz, NCCL_TYPE,
                            problem->rank - problem->npx - 1, nccl_comm, cuda_stream));
    }

    // Exchange front if available
    if (problem->pz > 0) {
        CHECK_NCCL(ncclRecv(halo->front_recv_buff_d, nx * ny, NCCL_TYPE,
                            problem->rank - problem->npx * problem->npy, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->front_send_buff_d, nx * ny, NCCL_TYPE,
                            problem->rank - problem->npx * problem->npy, nccl_comm, cuda_stream));
    }

    // Exchange back if available
    if (problem->pz < problem->npz - 1) {
        CHECK_NCCL(ncclRecv(halo->back_recv_buff_d, nx * ny, NCCL_TYPE,
                            problem->rank + problem->npx * problem->npy, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->back_send_buff_d, nx * ny, NCCL_TYPE,
                            problem->rank + problem->npx * problem->npy, nccl_comm, cuda_stream));
    }

    // Exchange front-north if available
    if (problem->py > 0 && problem->pz > 0) {
        CHECK_NCCL(ncclRecv(halo->front_north_recv_buff_d, nx, NCCL_TYPE,
                            problem->rank - problem->npx * (problem->npy + 1), nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->front_north_send_buff_d, nx, NCCL_TYPE,
                            problem->rank - problem->npx * (problem->npy + 1), nccl_comm, cuda_stream));
    }

    // Exchange front-east if available
    if (problem->px < problem->npx - 1 && problem->pz > 0) {
        CHECK_NCCL(ncclRecv(halo->front_east_recv_buff_d, ny, NCCL_TYPE,
                            problem->rank - problem->npx * problem->npy + 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->front_east_send_buff_d, ny, NCCL_TYPE,
                            problem->rank - problem->npx * problem->npy + 1, nccl_comm, cuda_stream));
    }

    // Exchange front-south if available
    if (problem->py < problem->npy - 1 && problem->pz > 0) {
        CHECK_NCCL(ncclRecv(halo->front_south_recv_buff_d, nx, NCCL_TYPE,
                            problem->rank - problem->npx * (problem->npy - 1), nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->front_south_send_buff_d, nx, NCCL_TYPE,
                            problem->rank - problem->npx * (problem->npy - 1), nccl_comm, cuda_stream));
    }

    // Exchange front-west if available
    if (problem->px > 0 && problem->pz > 0) {
        CHECK_NCCL(ncclRecv(halo->front_west_recv_buff_d, ny, NCCL_TYPE,
                            problem->rank - problem->npx * problem->npy - 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->front_west_send_buff_d, ny, NCCL_TYPE,
                            problem->rank - problem->npx * problem->npy - 1, nccl_comm, cuda_stream));
    }

    // Exchange back-north if available
    if (problem->py > 0 && problem->pz < problem->npz - 1) {
        CHECK_NCCL(ncclRecv(halo->back_north_recv_buff_d, nx, NCCL_TYPE,
                            problem->rank + problem->npx * (problem->npy - 1), nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->back_north_send_buff_d, nx, NCCL_TYPE,
                            problem->rank + problem->npx * (problem->npy - 1), nccl_comm, cuda_stream));
    }

    // Exchange back-east if available
    if (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1) {
        CHECK_NCCL(ncclRecv(halo->back_east_recv_buff_d, ny, NCCL_TYPE,
                            problem->rank + problem->npx * problem->npy + 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->back_east_send_buff_d, ny, NCCL_TYPE,
                            problem->rank + problem->npx * problem->npy + 1, nccl_comm, cuda_stream));
    }

    // Exchange back-south if available
    if (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1) {
        CHECK_NCCL(ncclRecv(halo->back_south_recv_buff_d, nx, NCCL_TYPE,
                            problem->rank + problem->npx * (problem->npy + 1), nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->back_south_send_buff_d, nx, NCCL_TYPE,
                            problem->rank + problem->npx * (problem->npy + 1), nccl_comm, cuda_stream));
    }

    // Exchange back-west if available
    if (problem->px > 0 && problem->pz < problem->npz - 1) {
        CHECK_NCCL(ncclRecv(halo->back_west_recv_buff_d, ny, NCCL_TYPE,
                            problem->rank + problem->npx * problem->npy - 1, nccl_comm, cuda_stream));
        CHECK_NCCL(ncclSend(halo->back_west_send_buff_d, ny, NCCL_TYPE,
                            problem->rank + problem->npx * problem->npy - 1, nccl_comm, cuda_stream));
    }

    // Exchange front corners if available
    if (problem->pz > 0) {
        // front-ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_NCCL(ncclRecv(halo->front_ne_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy + 1) + 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->front_ne_send_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy + 1) + 1, nccl_comm, cuda_stream));
        }
        // front-se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_NCCL(ncclRecv(halo->front_se_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy - 1) + 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->front_se_send_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy - 1) + 1, nccl_comm, cuda_stream));
        }
        // front-sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_NCCL(ncclRecv(halo->front_sw_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy - 1) - 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->front_sw_send_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy - 1) - 1, nccl_comm, cuda_stream));
        }
        // front-nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_NCCL(ncclRecv(halo->front_nw_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy + 1) - 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->front_nw_send_buff_d, 1, NCCL_TYPE,
                                problem->rank - problem->npx * (problem->npy + 1) - 1, nccl_comm, cuda_stream));
        }
    }

    // Exchange back corners if available
    if (problem->pz < problem->npz - 1) {
        // back-ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_NCCL(ncclRecv(halo->back_ne_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy - 1) + 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->back_ne_send_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy - 1) + 1, nccl_comm, cuda_stream));
        }
        // back-se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_NCCL(ncclRecv(halo->back_se_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy + 1) + 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->back_se_send_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy + 1) + 1, nccl_comm, cuda_stream));
        }
        // back-sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_NCCL(ncclRecv(halo->back_sw_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy + 1) - 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->back_sw_send_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy + 1) - 1, nccl_comm, cuda_stream));
        }
        // back-nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_NCCL(ncclRecv(halo->back_nw_recv_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy - 1) - 1, nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->back_nw_send_buff_d, 1, NCCL_TYPE,
                                problem->rank + problem->npx * (problem->npy - 1) - 1, nccl_comm, cuda_stream));
        }
    }

    ncclGroupEnd();
}

void injectData(Halo *halo, Problem *problem){
    int dimx = halo->dimx;
    int dimy = halo->dimy;
    int dimz = halo->dimz;
    int nx = halo->nx;
    int ny = halo->ny;
    int nz = halo->nz;
    DataType *x_d = halo->x_d;

    // Inject north if got north
    if (problem->py > 0) {
        inject_horizontal_plane_to_GPU(x_d, halo->north_recv_buff_d, 1, 0, 1, nx, nz, dimx, dimy, dimz);
    }

    // Inject east if got east
    if (problem->px < problem->npx - 1) {
        inject_vertical_plane_to_GPU(x_d, halo->east_recv_buff_d, dimx - 1, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Inject south if got south
    if (problem->py < problem->npy - 1) {
        inject_horizontal_plane_to_GPU(x_d, halo->south_recv_buff_d, 1, dimy - 1, 1, nx, nz, dimx, dimy, dimz);
    }

    // Inject west if got west
    if (problem->px > 0) {
        inject_vertical_plane_to_GPU(x_d, halo->west_recv_buff_d, 0, 1, 1, ny, nz, dimx, dimy, dimz);
    }

    // Inject ne if got ne
    if (problem->px < problem->npx - 1 && problem->py > 0) {
        inject_edge_Z_to_GPU(x_d, halo->ne_recv_buff_d, dimx - 1, 0, 1, nz, dimx, dimy, dimz);
    }

    // Inject se if got se
    if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
        inject_edge_Z_to_GPU(x_d, halo->se_recv_buff_d, dimx - 1, dimy - 1, 1, nz, dimx, dimy, dimz);
    }

    // Inject sw if got sw
    if (problem->px > 0 && problem->py < problem->npy - 1) {
        inject_edge_Z_to_GPU(x_d, halo->sw_recv_buff_d, 0, dimy - 1, 1, nz, dimx, dimy, dimz);
    }

    // Inject nw if got nw
    if (problem->py > 0 && problem->px > 0) {
        inject_edge_Z_to_GPU(x_d, halo->nw_recv_buff_d, 0, 0, 1, nz, dimx, dimy, dimz);
    }

    // Inject front if got front
    if (problem->pz > 0) {
        inject_frontal_plane_to_GPU(x_d, halo->front_recv_buff_d, 1, 1, 0, nx, ny, dimx, dimy, dimz);
    }

    // Inject back if got back
    if (problem->pz < problem->npz - 1) {
        inject_frontal_plane_to_GPU(x_d, halo->back_recv_buff_d, 1, 1, dimz - 1, nx, ny, dimx, dimy, dimz);
    }

    // Inject front_north if got front_north
    if (problem->py > 0 && problem->pz > 0) {
        inject_edge_X_to_GPU(x_d, halo->front_north_recv_buff_d, 1, 0, 0, nx, dimx, dimy, dimz);
    }

    // Inject front_east if got front_east
    if (problem->px < problem->npx - 1 && problem->pz > 0) {
        inject_edge_Y_to_GPU(x_d, halo->front_east_recv_buff_d, dimx - 1, 1, 0, ny, dimx, dimy, dimz);
    }

    // Inject front_south if got front_south
    if (problem->py < problem->npy - 1 && problem->pz > 0) {
        inject_edge_X_to_GPU(x_d, halo->front_south_recv_buff_d, 1, dimy - 1, 0, nx, dimx, dimy, dimz);
    }

    // Inject front_west if got front_west
    if (problem->px > 0 && problem->pz > 0) {
        inject_edge_Y_to_GPU(x_d, halo->front_west_recv_buff_d, 0, 1, 0, ny, dimx, dimy, dimz);
    }

    // Inject back_north if got back_north
    if (problem->py > 0 && problem->pz < problem->npz - 1) {
        inject_edge_X_to_GPU(x_d, halo->back_north_recv_buff_d, 1, 0, dimz - 1, nx, dimx, dimy, dimz);
    }

    // Inject back_east if got back_east
    if (problem->px < problem->npx - 1 && problem->pz < problem->npz - 1) {
        inject_edge_Y_to_GPU(x_d, halo->back_east_recv_buff_d, dimx - 1, 1, dimz - 1, ny, dimx, dimy, dimz);
    }

    // Inject back_south if got back_south
    if (problem->py < problem->npy - 1 && problem->pz < problem->npz - 1) {
        inject_edge_X_to_GPU(x_d, halo->back_south_recv_buff_d, 1, dimy - 1, dimz - 1, nx, dimx, dimy, dimz);
    }

    // Inject back_west if got back_west
    if (problem->px > 0 && problem->pz < problem->npz - 1) {
        inject_edge_Y_to_GPU(x_d, halo->back_west_recv_buff_d, 0, 1, dimz - 1, ny, dimx, dimy, dimz);
    }

    // Inject front corners
    if (problem->pz > 0) {
        // Inject front_ne if got front_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1, halo->front_ne_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject front_se if got front_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy - 1, halo->front_se_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject front_sw if got front_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d + (dimy - 1) * dimx, halo->front_sw_recv_buff_d, 
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject front_nw if got front_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d, halo->front_nw_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }

    // Inject back corners
    if (problem->pz < problem->npz - 1) {
        // Inject back_ne if got back_ne
        if (problem->py > 0 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + dimx * dimy * (dimz - 1), halo->back_ne_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject back_se if got back_se
        if (problem->py < problem->npy - 1 && problem->px < problem->npx - 1) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx - 1 + (dimy - 1) * dimx + dimx * dimy * (dimz - 1), halo->back_se_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject back_sw if got back_sw
        if (problem->py < problem->npy - 1 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx * (dimy - 1) + dimx * dimy * (dimz - 1), halo->back_sw_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
        // Inject back_nw if got back_nw
        if (problem->py > 0 && problem->px > 0) {
            CHECK_CUDA(cudaMemcpy(x_d + dimx * dimy * (dimz - 1), halo->back_nw_recv_buff_d,
                                    sizeof(DataType), cudaMemcpyHostToDevice));
        }
    }
}

// Copies to host and back to device, no device-to-device copy
// TODO: Replace malloc per exchange with malloc once at beginning
template <typename T>
void NCCL_Implementation<T>::ExchangeHaloNCCL(Halo *halo, Problem *problem) {

    //ensure computation is done
    CHECK_CUDA(cudaDeviceSynchronize());

    //Fill the send buffers
    extractData(halo, problem);
    
    // Wait until send buffers are ready
    CHECK_CUDA(cudaDeviceSynchronize());

    // Now we can start the NCCL communication
    sendRecvData(halo, problem, this->nccl_comm, this->cuda_stream);

    // synchronizing on CUDA stream to complete NCCL communication
    CHECK_CUDA(cudaStreamSynchronize(this->cuda_stream));

    // Now that we received all data, we can inject it back to the halo
    injectData(halo, problem);

    // Wait until received data is injected
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
void NCCL_Implementation<T>::finalize_comm_NCCL(Problem *problem){

    //wait for all NCCL operations to finish
    CHECK_CUDA(cudaStreamSynchronize(this->cuda_stream));

    //finalizing NCCL
    ncclCommDestroy(this->nccl_comm);

    //finalizing MPI
    MPI_Finalize();
}

// Explicit template instantiation
template class NCCL_Implementation<DataType>;