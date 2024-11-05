#include "benchmark.hpp"
// #include "TimingLib/timer.hpp"

// these function calls the abstract function the required number of times and records the time

// again we have method overloading for different matrix types

// this SPMV supports CSR matrixes
void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d, double * y_d
    ){

    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
    // we always test against cusparse$
        cuSparse_Implementation<double> baseline;
        bool test_failed = !test_SPMV(
            baseline, implementation,
            A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d);
        if (test_failed){
            num_iterations = 0;
        }
    }


    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_SPMV(
            A,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            x_d, y_d
        );
        timer.stopTimer("compute_SPMV");
    }
}

// this SPMV supports banded matrixes which requires CSR for metadata and testing
void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    banded_Matrix<double> & A,
    double * banded_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * x_d, double * y_d
    ){

    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
    // we always test against cusparse
        cuSparse_Implementation<double> baseline;
        sparse_CSR_Matrix<double> sparse_CSR_A;
        sparse_CSR_A.sparse_CSR_Matrix_from_banded(A); 

        int num_rows = sparse_CSR_A.get_num_rows();
        int num_cols = sparse_CSR_A.get_num_cols();
        int nnz = sparse_CSR_A.get_nnz();

        int * A_row_ptr_data = sparse_CSR_A.get_row_ptr().data();
        int * A_col_idx_data = sparse_CSR_A.get_col_idx().data();
        double * A_values_data = sparse_CSR_A.get_values().data();

        int * A_row_ptr_d;
        int * A_col_idx_d;
        double * A_values_d;

        CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
        // test the SPMV function
        bool test_failed = !test_SPMV(
            baseline, implementation,
            A,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            
            banded_A_d,
            num_rows, num_cols,
            num_bands,
            j_min_i_d,

            x_d
            );

        if (test_failed)
        {
            num_iterations = 0;
        }
    }


    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_SPMV(
            A,
            banded_A_d,
            num_rows, num_cols,
            num_bands,
            j_min_i_d,
            x_d, y_d
        );
        timer.stopTimer("compute_SPMV");
    }

}

// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d, double * y_d
    ){

    bench_SPMV(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
    // other functions to be benchmarked
}

// this version supports banded matrixes
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    banded_Matrix<double> & A, // we need to pass the CSR matrix for metadata and potential testing
    double * banded_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * x_d, double * y_d
    ){

    bench_SPMV(implementation, timer, A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, x_d, y_d);
    // other functions to be benchmarked
}

