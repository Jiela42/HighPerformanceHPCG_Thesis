#include "HPCG_versions/non_blocking_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

template <typename T>
Problem* non_blocking_mpi_Implementation<T>::init_comm_non_blocking_MPI(int argc, char *argv[], int npx, int npy, int npz, int nx, int ny, int nz){
    MPI_Init( &argc , &argv );
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    Problem *problem = (Problem *)malloc(sizeof(Problem));
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, problem);

    InitGPU(problem);

    return problem;
}

// Copies to host and back to device, no device-to-device copy
// TODO: Introduce cudaStreams
template <typename T>
void non_blocking_mpi_Implementation<T>::ExchangeHaloNonBlockingMPI(Halo *halo, Problem *problem) {

    //extract the data into send buffers on the GPU
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, halo->send_buff_h[i], &problem->extraction_ghost_cells[i]);
        }
    }

    // We now need twice as many requests per exchange
    MPI_Request requests[52];
    int msg_count = 0;
    
    //post receive calls
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Irecv(halo->recv_buff_h[i], problem->count_exchange[i], MPIDataType, problem->neighbors[i], 0, MPI_COMM_WORLD, &requests[msg_count]));
            msg_count++;
        }
    }

    //wait for extraction to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    //post send calls
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Isend(halo->send_buff_h[i], problem->count_exchange[i], MPIDataType, problem->neighbors[i], 0, MPI_COMM_WORLD, &requests[msg_count]));
            msg_count++;
        }
    }

    // Wait until all exchanges are done
    CHECK_MPI(MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE));

    // Now that we received all data, we can inject it back to the halo
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, halo->recv_buff_h[i], &problem->injection_ghost_cells[i]);
        }
    }

    //wait for injection to be done
    CHECK_CUDA(cudaDeviceSynchronize());

}

template <typename T>
void non_blocking_mpi_Implementation<T>::finalize_comm_non_blocking_MPI(Problem *problem){
    MPI_Finalize();
}

// Explicit template instantiation
template class non_blocking_mpi_Implementation<DataType>;