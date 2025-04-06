#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

template <typename T>
Problem* blocking_mpi_Implementation<T>::init_comm_blocking_MPI(int argc, char *argv[], int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz){
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    Problem *problem = (Problem *)malloc(sizeof(Problem));
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, problem);

    InitGPU(problem);

    return problem;
}

// Copies to host and back to device, no device-to-device copy
// TODO: Replace malloc per exchange with malloc once at beginning
template <typename T>
void blocking_mpi_Implementation<T>::ExchangeHaloBlockingMPI(Halo *halo, Problem *problem) {

    //extract the data into send buffers on the GPU
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, i, &(problem->extraction_ghost_cells[i]), 1);
        }
    }
    
    
    //wait for extraction to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    //do blocking SendRecv
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Sendrecv(halo->send_buff_h[i], problem->count_exchange[i], MPI_DOUBLE,
                                    problem->neighbors[i], 0,
                                    halo->recv_buff_h[i], problem->count_exchange[i], MPI_DOUBLE,
                                    problem->neighbors[i], 0,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
    }

    // Now that we received all data, we can inject it back to the halo
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, i, &(problem->injection_ghost_cells[i]), 1);
        }
    }

    //wait for injection to be done
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
void blocking_mpi_Implementation<T>::finalize_comm_blocking_MPI(Problem *problem){
    MPI_Finalize();
}

// Explicit template instantiation
template class blocking_mpi_Implementation<DataType>;