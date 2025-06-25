// Implements blocking MPI-based halo exchange using host buffers and synchronous communication.

#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

template <typename T>
Problem* blocking_mpi_Implementation<T>::init_comm_blocking_MPI(int argc, char *argv[], int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, bool initMPI){
    // Initialize MPI environment and set up GPU device for the problem
    if(initMPI){
        MPI_Init(&argc, &argv);
    }
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    Problem *problem = (Problem *)malloc(sizeof(Problem));
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, problem);

    InitGPU(problem);

    return problem;
}

template <typename T>
void blocking_mpi_Implementation<T>::ExchangeHaloBlockingMPI(Halo *halo, Problem *problem) {
    // Perform blocking halo exchange using MPI with host buffers

    // Extract the data into send buffers on the GPU
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, i, &(problem->extraction_ghost_cells[i]), 1);
        }
    }
    
    // Wait for extraction to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Do blocking SendRecv communication
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Sendrecv(halo->send_buff_h[i], problem->count_exchange[i], MPI_DOUBLE,
                                    problem->neighbors[i], 0,
                                    halo->recv_buff_h[i], problem->count_exchange[i], MPI_DOUBLE,
                                    problem->neighbors[i], 0,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
    }

    // Now that we received all data, inject it back to the halo
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, i, &(problem->injection_ghost_cells[i]), 1);
        }
    }

    // Wait for injection to be done
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
void blocking_mpi_Implementation<T>::finalize_comm_blocking_MPI(Problem *problem){
    // Finalize the MPI environment
    MPI_Finalize();
}

// Explicit template instantiation
template class blocking_mpi_Implementation<DataType>;