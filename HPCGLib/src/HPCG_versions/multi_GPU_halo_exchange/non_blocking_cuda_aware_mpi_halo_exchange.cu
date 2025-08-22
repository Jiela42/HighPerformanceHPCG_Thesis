// Implements non-blocking CUDA-aware MPI halo exchange using device buffers and asynchronous MPI calls.

#include "HPCG_versions/non_blocking_cuda_aware_mpi_halo_exchange.cuh"
#include "UtilLib/utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

template <typename T>
Problem* non_blocking_cuda_aware_mpi_Implementation<T>::init_comm_non_blocking_cuda_aware_MPI(int argc, char *argv[], int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, bool initMPI) {
    // Initialize MPI environment and select GPU device
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

template <typename T>
void non_blocking_cuda_aware_mpi_Implementation<T>::ExchangeHaloNonBlockingCudaAwareMPI(Halo *halo, Problem *problem) {
    // Perform halo exchange using non-blocking CUDA-aware MPI

    cudaStreamSynchronize(*(halo->streams[26])); // Wait for border computation to finish

    // Extract the data into send buffers on the GPU
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->extraction_functions[i])(halo, i, &problem->extraction_ghost_cells[i], 0);
        }
    }

    // We now need twice as many requests per exchange
    MPI_Request requests[52];
    int msg_count = 0;
    
    // Post receive calls
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            CHECK_MPI(MPI_Irecv(halo->recv_buff_d[i], problem->count_exchange[i], MPIDataType, problem->neighbors[i], 0, MPI_COMM_WORLD, &requests[msg_count]));
            msg_count++;
        }
    }

    // Post send calls
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            cudaStreamSynchronize(*(halo->streams[i])); // Ensure the extraction is done before sending
            CHECK_MPI(MPI_Isend(halo->send_buff_d[i], problem->count_exchange[i], MPIDataType, problem->neighbors[i], 0, MPI_COMM_WORLD, &requests[msg_count]));
            msg_count++;
        }
    }

    // Wait until all exchanges are done
    CHECK_MPI(MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE));

    // Now that we received all data, inject it back to the halo
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            (*problem->injection_functions[i])(halo, i, &problem->injection_ghost_cells[i], 0);
        }
    }

    // Wait for injection to be done
    for(int i = 0; i<NUMBER_NEIGHBORS; i++){
        if(problem->neighbors_mask[i]){
            cudaStreamSynchronize(*(halo->streams[i]));
        }
    }

}

template <typename T>
void non_blocking_cuda_aware_mpi_Implementation<T>::finalize_comm_non_blocking_cuda_aware_MPI(Problem *problem){
    // Finalize MPI environment
    MPI_Finalize();
}

void initialize_COO_comm(Problem *problem, striped_partial_Matrix<DataType> &A_local) {
    if(A_local.initialized_COO_comm) {
        // If COO communication is already initialized, return early
        return;
    }
    //communicate how many values to recv from each rank
    MPI_Request requests[problem->size*2]; // double the size for send and recv
    int msg_count = 0;
    //first communicate how many values to send to each rank
    //post receives for COO communication
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank) continue; // skip self
        CHECK_MPI(MPI_Irecv(&A_local.send_per_rank_COO[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[msg_count])); //ask how many values to send
        msg_count++;
    }
    //post sends for COO communication
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank) continue; // skip self
        CHECK_MPI(MPI_Isend(&A_local.req_per_rank_COO[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[msg_count])); //name how many values to send
        msg_count++;
    }
    //wait for all requests to finish
    CHECK_MPI(MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE));

    //count how many values to receive and send in total
    for(int i = 0; i < problem->size; i++) {
        A_local.total_to_send_COO += A_local.send_per_rank_COO[i];
        A_local.total_to_recv_COO += A_local.req_per_rank_COO[i];
    }
    //allocate memory for the pointers to the indices to send and receive
    global_int_t *mem_idx_to_send_COO = (global_int_t *)malloc(A_local.total_to_send_COO * sizeof(global_int_t));
    global_int_t *mem_idx_to_recv_COO = (global_int_t *)malloc(A_local.total_to_recv_COO * sizeof(global_int_t));
    DataType *mem_send_buff_COO = (DataType *)malloc(A_local.total_to_send_COO * sizeof(DataType));
    DataType *mem_recv_buff_COO = (DataType *)malloc(A_local.total_to_recv_COO * sizeof(DataType));
    for(int i = 0; i < problem->size; i++) {
        A_local.ptr_idx_to_send_COO_h[i] = mem_idx_to_send_COO;
        mem_idx_to_send_COO += A_local.send_per_rank_COO[i];
        A_local.ptr_idx_to_recv_COO_h[i] = mem_idx_to_recv_COO;
        mem_idx_to_recv_COO += A_local.req_per_rank_COO[i];
        A_local.ptr_send_buff_COO_h[i] = mem_send_buff_COO;
        mem_send_buff_COO += A_local.send_per_rank_COO[i];
        A_local.ptr_recv_buff_COO_h[i] = mem_recv_buff_COO;
        mem_recv_buff_COO += A_local.req_per_rank_COO[i];
    }

    //figure out which indices to receive per rank
    for(int i = 0; i < problem->size; i++) {
        if(A_local.req_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to receive
        A_local.ptr_idx_to_recv_COO_h[i] = (global_int_t *)malloc(A_local.req_per_rank_COO[i] * sizeof(global_int_t));
    }
    
    // Initialize count_per_rank to zero
    int count_per_rank[problem->size];
    memset(count_per_rank, 0, sizeof(int) * problem->size);
    
    for(int i = 0; i < A_local.nnz_COO; i++) {
        int rank = A_local.col_to_rank_COO[i];
        A_local.ptr_idx_to_recv_COO_h[rank][count_per_rank[rank]] = A_local.col_COO[i];
        count_per_rank[rank]++;
    }

    //exchange idxes to send and receive
    msg_count = 0;
    //post receives for COO communication
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank || A_local.send_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to send
        CHECK_MPI(MPI_Irecv(A_local.ptr_idx_to_send_COO_h[i], A_local.send_per_rank_COO[i], MPI_LONG, i, 0, MPI_COMM_WORLD, &requests[msg_count])); //ask which values to send
        msg_count++;
    }
    //post sends for COO communication
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank || A_local.req_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to receive
        CHECK_MPI(MPI_Isend(A_local.ptr_idx_to_recv_COO_h[i], A_local.req_per_rank_COO[i], MPI_LONG, i, 0, MPI_COMM_WORLD, &requests[msg_count])); //name which values to receive
        msg_count++;
    }
    //wait for all requests to finish
    CHECK_MPI(MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE));
    if(problem->rank == 0) {
        printf("COO communication initialized with %d ranks.\n", problem->size);
    }
    A_local.initialized_COO_comm = true; // Set the flag to indicate that COO communication has been initialized

    //for debugging: print row, col, data, then col_to_rank, then ptr_idx_to_send_COO_h and ptr_idx_to_recv_COO_h
    printf("Rank = %d\n", problem->rank);
    printf("Rank = %d, nnz_COO = %ld\n", problem->rank, A_local.nnz_COO);
    for(int i = 0; i < A_local.nnz_COO; i++) {
        printf("Rank = %d, COO[%d] = (%ld, %ld, %f)\n", problem->rank, i, A_local.row_COO[i], A_local.col_COO[i], A_local.data_COO[i]);
    }
    printf("Rank = %d, col_to_rank_COO:\n", problem->rank);
    for(int i = 0; i < A_local.nnz_COO; i++) {
        printf("Rank = %d, col_to_rank_COO[%d] = %d\n", problem->rank, i, A_local.col_to_rank_COO[i]);
    }
    printf("Rank = %d, ptr_idx_to_send_COO_h:\n", problem->rank);
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank || A_local.send_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to send
        printf("Rank = %d, ptr_idx_to_send_COO_h[%d]: ", problem->rank, i);
        for(int j = 0; j < A_local.send_per_rank_COO[i]; j++) {
            printf("%ld ", A_local.ptr_idx_to_send_COO_h[i][j]);
        }
        printf("\n");
    }
    printf("Rank = %d, ptr_idx_to_recv_COO_h:\n", problem->rank);
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank || A_local.req_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to receive
        printf("Rank = %d, ptr_idx_to_recv_COO_h[%d]: ", problem->rank, i);
        for(int j = 0; j < A_local.req_per_rank_COO[i]; j++) {
            printf("%ld ", A_local.ptr_idx_to_recv_COO_h[i][j]);
        }
        printf("\n");
    }
}

// extract the data named in the idx buffers and receive changed values
void exchange_COO_data(Halo *halo, Problem *problem, striped_partial_Matrix<DataType> &A_local) {
    if(!A_local.initialized_COO_comm) {
        printf("COO communication not initialized. Call initialize_COO_comm first.\n");
        return;
    }
    //extract the data to send
    //extract_COO_data(halo, problem, A_local);
    //exchange the data
    MPI_Request requests[problem->size*2]; // double the size for send and recv
    int msg_count = 0;
    //post receives for COO communication
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank || A_local.req_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to receive
        CHECK_MPI(MPI_Irecv(A_local.ptr_recv_buff_COO_h[i], A_local.req_per_rank_COO[i], MPIDataType, i, 0, MPI_COMM_WORLD, &requests[msg_count])); //ask for the values
        msg_count++;
    }
    //wait for extraction to be done before sending
    CHECK_CUDA(cudaStreamSynchronize(*(halo->streams[28]))); // Ensure the extraction is done before sending
    //post sends for COO communication
    for(int i = 0; i < problem->size; i++) {
        if(i == problem->rank || A_local.send_per_rank_COO[i] == 0) continue; // skip self and ranks with no data to send
        CHECK_MPI(MPI_Isend(A_local.ptr_send_buff_COO_h[i], A_local.send_per_rank_COO[i], MPIDataType, i, 0, MPI_COMM_WORLD, &requests[msg_count])); //send the values
        msg_count++;
    }
    //wait for all requests to finish
    CHECK_MPI(MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE));

}
// Explicit template instantiation
template class non_blocking_cuda_aware_mpi_Implementation<DataType>;