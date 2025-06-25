// Implements non-blocking host-only MPI halo exchange using host buffers and asynchronous MPI calls.

#include "HPCG_versions/non_blocking_host_only_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

template <typename T>
Problem* non_blocking_host_only_mpi_Implementation<T>::init_comm_non_blocking_host_only_MPI(int argc, char *argv[], int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, bool initMPI) {
    // Initialize MPI environment and select GPU device for this rank
    if(initMPI){
        CHECK_MPI(MPI_Init(&argc, &argv));
    }
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    Problem *problem = (Problem *)malloc(sizeof(Problem));
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, problem);

    InitGPU(problem);

    return problem;
}

// TODO: Introduce cudaStreams
template <typename T>
void non_blocking_host_only_mpi_Implementation<T>::ExchangeHaloNonBlockingHostOnlyMPI(Halo *halo, Problem *problem) {
    // Perform halo data exchange using non-blocking MPI with host buffers

    // Extract the data into send buffers on the GPU
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, i, &problem->extraction_ghost_cells[i], 1);
        }
    }

    // We now need twice as many requests per exchange
    MPI_Request requests[52];
    int msg_count = 0;
    
    // Post receive calls for incoming halo data
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Irecv(halo->recv_buff_h[i], problem->count_exchange[i], MPIDataType, problem->neighbors[i], 0, MPI_COMM_WORLD, &requests[msg_count]));
            msg_count++;
        }
    }

    // Wait for extraction to finish before sending
    CHECK_CUDA(cudaDeviceSynchronize());

    // Post send calls for outgoing halo data
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Isend(halo->send_buff_h[i], problem->count_exchange[i], MPIDataType, problem->neighbors[i], 0, MPI_COMM_WORLD, &requests[msg_count]));
            msg_count++;
        }
    }

    // Wait until all exchanges are done
    CHECK_MPI(MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE));

    // Inject the received data back into the halo region on the device
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, i, &problem->injection_ghost_cells[i], 1);
        }
    }

    // Wait for injection to be done
    CHECK_CUDA(cudaDeviceSynchronize());

}

template <typename T>
void non_blocking_host_only_mpi_Implementation<T>::finalize_comm_non_blocking_host_only_MPI(Problem *problem){
    // Finalize the MPI environment
    MPI_Finalize();
}

// Explicit template instantiation
template class non_blocking_host_only_mpi_Implementation<DataType>;