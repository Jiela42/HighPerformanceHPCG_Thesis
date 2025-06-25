// Implements NCCL-based halo exchange using device buffers and GPU-aware collectives.

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
Problem* NCCL_Implementation<T>::init_comm_NCCL(int argc, char *argv[], int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, bool initMPI) {
    
    // Initialize MPI and determine rank and size
    if(initMPI){
        CHECK_MPI(MPI_Init(&argc, &argv));
    }
    int size, rank, localRank = 0;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // Calculate localRank for GPU device selection based on host hash
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

    // Select GPU device based on localRank if multiple devices are available
    if(deviceCount > 1){
        CHECK_CUDA(cudaSetDevice(localRank));
    }

    CHECK_CUDA(cudaStreamCreate(&this->cuda_stream));

    // Get NCCL unique ID at rank 0 and broadcast it to all ranks
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    CHECK_MPI(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL communicator
    CHECK_NCCL(ncclCommInitRank(&this->nccl_comm, size, id, rank));

    // Initialize the problem data structure
    Problem *problem = (Problem *)malloc(sizeof(Problem));
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, problem);

    return problem;
}

// TODO: Use seperate cudaStreams
template <typename T>
void NCCL_Implementation<T>::ExchangeHaloNCCL(Halo *halo, Problem *problem) {
    // Perform halo exchange using NCCL with device buffers

    // Ensure all previous computations are complete
    CHECK_CUDA(cudaDeviceSynchronize());

    // Extract data to send buffers from halo regions
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, i, &problem->extraction_ghost_cells[i], 0);
        }
    }
    
    // Synchronize to ensure send buffers are ready
    CHECK_CUDA(cudaDeviceSynchronize());

    // Start NCCL communication group
    ncclGroupStart();
    
    // Post receives and sends for all neighbors
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_NCCL(ncclRecv(halo->recv_buff_d[i], problem->count_exchange[i], NCCL_TYPE,
                        problem->neighbors[i], this->nccl_comm, cuda_stream));
            CHECK_NCCL(ncclSend(halo->send_buff_d[i], problem->count_exchange[i], NCCL_TYPE,
                        problem->neighbors[i], this->nccl_comm, cuda_stream));
        }
    }

    // End NCCL communication group
    ncclGroupEnd();

    // Synchronize CUDA stream to ensure NCCL communication completes
    CHECK_CUDA(cudaStreamSynchronize(this->cuda_stream));

    // Inject received data back into halo regions
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, i, &problem->injection_ghost_cells[i], 0);
        }
    }

    // Synchronize to ensure injection is complete
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
void NCCL_Implementation<T>::finalize_comm_NCCL(Problem *problem){
    // Finalize NCCL communication and MPI environment

    // Wait for all NCCL operations to complete
    CHECK_CUDA(cudaStreamSynchronize(this->cuda_stream));

    // Destroy NCCL communicator
    ncclCommDestroy(this->nccl_comm);

    // Finalize MPI
    MPI_Finalize();
}

// Explicit template instantiation
template class NCCL_Implementation<DataType>;