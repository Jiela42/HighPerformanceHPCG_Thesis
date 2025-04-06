#include "HPCG_versions/nccl_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

#include "nccl.h"

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
Problem* NCCL_Implementation<T>::init_comm_NCCL(int argc, char *argv[], int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz) {
    
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

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    //picking a GPU based on localRank
    if(deviceCount > 1){
        CHECK_CUDA(cudaSetDevice(localRank));
    }

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

// Copies to host and back to device, no device-to-device copy
// TODO: Use seperate cudaStreams
template <typename T>
void NCCL_Implementation<T>::ExchangeHaloNCCL(Halo *halo, Problem *problem) {

    //ensure computation is done
    CHECK_CUDA(cudaDeviceSynchronize());

    //Fill the send buffers
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, i, &problem->extraction_ghost_cells[i], 0);
        }
    }
    
    // Wait until send buffers are ready
    CHECK_CUDA(cudaDeviceSynchronize());

    // Now we can start the NCCL communication
    ncclGroupStart();
    
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_NCCL(ncclRecv(halo->recv_buff_d[i], problem->count_exchange[i], NCCL_TYPE,
                        problem->neighbors[i], this->nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->send_buff_d[i], problem->count_exchange[i], NCCL_TYPE,
                        problem->neighbors[i], this->nccl_comm, cuda_stream));
        }
    }

    ncclGroupEnd();

    // synchronizing on CUDA stream to complete NCCL communication
    CHECK_CUDA(cudaStreamSynchronize(this->cuda_stream));

    // Now that we received all data, we can inject it back to the halo
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, i, &problem->injection_ghost_cells[i], 0);
        }
    }

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