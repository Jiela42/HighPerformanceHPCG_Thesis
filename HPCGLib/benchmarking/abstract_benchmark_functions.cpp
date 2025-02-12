#include "benchmark.hpp"
#include <sstream>
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
    )
{

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<double> y(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(y_d, y.data(), A.get_num_rows() * sizeof(double), cudaMemcpyHostToDevice));

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

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(y.data(), y_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));
}

// this SPMV supports striped matrixes which requires CSR for metadata and testing
void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A,
    double * striped_A_d,
    int num_rows, int num_cols,
    int num_stripes,
    int * j_min_i_d,
    double * x_d, double * y_d
    ){

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<double> y(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(y_d, y.data(), A.get_num_rows() * sizeof(double), cudaMemcpyHostToDevice));

    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
    // we always test against cusparse
        cuSparse_Implementation<double> baseline;
        sparse_CSR_Matrix<double> sparse_CSR_A;
        sparse_CSR_A.sparse_CSR_Matrix_from_striped(A); 

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
            
            striped_A_d,
            num_rows, num_cols,
            num_stripes,
            j_min_i_d,

            x_d
            );

        // free the memory
        cudaFree(A_row_ptr_d);
        cudaFree(A_col_idx_d);
        cudaFree(A_values_d);

        if (test_failed)
        {
            num_iterations = 0;
        }
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_SPMV(
            A,
            striped_A_d,
            num_rows, num_cols,
            num_stripes,
            j_min_i_d,
            x_d, y_d
        );
        timer.stopTimer("compute_SPMV");
    }

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(y.data(), y_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));
}

void bench_Dot(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d, double * result_d
    ){
    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
        // note that for the dot product the cuSparse implementation is an instanciation of warp reduction. ehem.
        cuSparse_Implementation<double> baseline;
        bool test_failed = !test_Dot(
            baseline, implementation,
            A,
            x_d, y_d
        );

        if (test_failed){
            num_iterations = 0;
        }            
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_Dot(
            A,
            x_d, y_d, result_d
        );
        timer.stopTimer("compute_Dot");
    }
    
}

void bench_Dot(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d, double * result_d
    ){
    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
        // note that for the dot product the cuSparse implementation is an instanciation of warp reduction. ehem.
        cuSparse_Implementation<double> baseline;
        bool test_failed = !test_Dot(
            implementation,
            A,
            x_d, y_d
        );

        if (test_failed){
            num_iterations = 0;
        }            
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_Dot(
            A,
            x_d, y_d, result_d
        );
        timer.stopTimer("compute_Dot");
    }
    
}


void bench_SymGS(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d, double * y_d
    )
{
    int num_iterations = implementation.getNumberOfIterations();

    // we need the following parameters to compute the norm
    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<double> x(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(x.data(), x_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));


    if (implementation.test_before_bench){
        cuSparse_Implementation<double> baseline;

        bool test_failed = !test_SymGS(
            baseline, implementation,
            A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
        if (test_failed){
            num_iterations = 0;
        }
    }

    for(int i = 0; i < num_iterations; i++){
        // always write the original x back into x_d
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_cols() * sizeof(double), cudaMemcpyHostToDevice));
        timer.startTimer();
        implementation.compute_SymGS(
            A,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            x_d, y_d
        );
        timer.stopTimer("compute_SymGS");
    }

    // greb da norm and store it in additional infos
    double norm = relative_residual_norm_for_SymGS(
        num_rows, num_cols,
        A_row_ptr_d, A_col_idx_d, A_values_d,
        x_d, y_d);

    std::ostringstream oss;
    oss << "RR Norm: " << norm;
    std::string norm_string = oss.str();
    timer.add_additional_parameters(norm_string);

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_rows() * sizeof(double), cudaMemcpyHostToDevice));
}

void bench_SymGS(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A,
    double * striped_A_d,
    int num_rows, int num_cols,
    int num_stripes,
    int * j_min_i_d,
    double * x_d, double * y_d
    )
{   
    int num_iterations = implementation.getNumberOfIterations();
    // std::cout << "benching symgs for " << num_iterations << " iterations" << std::endl;

    // x_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<double> x(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(x.data(), x_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));

    if(implementation.test_before_bench){
        sparse_CSR_Matrix<double> A_csr;
        A_csr.sparse_CSR_Matrix_from_striped(A);

        cuSparse_Implementation<double> baseline;

        int num_rows = A_csr.get_num_rows();
        int num_cols = A_csr.get_num_cols();
        int nnz = A_csr.get_nnz();

        int * A_row_ptr_data = A_csr.get_row_ptr().data();
        int * A_col_idx_data = A_csr.get_col_idx().data();
        double * A_values_data = A_csr.get_values().data();

        int * A_row_ptr_d;
        int * A_col_idx_d;
        double * A_values_d;

        CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
            

        bool test_failed = !test_SymGS(
            baseline, implementation,
            A,
            A_row_ptr_d, A_col_idx_d, A_values_d,

            striped_A_d,
            num_rows, num_cols,
            num_stripes,
            j_min_i_d,

            y_d);

        // now we need to free the memory
        CHECK_CUDA(cudaFree(A_row_ptr_d));
        CHECK_CUDA(cudaFree(A_col_idx_d));
        CHECK_CUDA(cudaFree(A_values_d));

        if (test_failed){
            num_iterations = 0;
        }
    }
        
    for (int i = 0; i < num_iterations; i++){
        // std::cout<< "Iteration: " << i << std::endl;
        // std::cout<< "Num iterations: " << num_iterations << std::endl;
        // copy original x into x_d
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_cols() * sizeof(double), cudaMemcpyHostToDevice));
        timer.startTimer();
        implementation.compute_SymGS( A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, x_d, y_d);
        timer.stopTimer("compute_SymGS");
    }

    // greb da norm and store it in additional infos
    double norm = relative_residual_norm_for_SymGS(
    num_rows, num_cols,
    num_stripes, j_min_i_d,
    striped_A_d,
    x_d, y_d);

    std::ostringstream oss;
    oss << "RR Norm: " << norm;
    std::string norm_string = oss.str();
    timer.add_additional_parameters(norm_string);


    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_rows() * sizeof(double), cudaMemcpyHostToDevice));
}

// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * a_d, double * b_d, // a & b are random vectors
    double * x_d, double * y_d // x & y are vectors as used in HPCG
    )
{
    if(implementation.SPMV_implemented){
        bench_SPMV(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, a_d, y_d);
    }
    if(implementation.SymGS_implemented){
        bench_SymGS(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
    }

    // bench_SPMV(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
    // bench_SymGS(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
    // other functions to be benchmarked
}

// this version supports striped matrixes
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A, // we need to pass the CSR matrix for metadata and potential testing
    double * striped_A_d,
    int num_rows, int num_cols,
    int num_stripes,
    int * j_min_i_d,
    double * a_d, double * b_d, // a & b are random vectors
    double * x_d, double * y_d, // x & y are vectors as used in HPCG
    double * result_d   // result is used for the dot product (it is a scalar)
    ){
      
    if(implementation.SPMV_implemented){
        bench_SPMV(implementation, timer, A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, a_d, y_d);
    }
    if(implementation.Dot_implemented){
        bench_Dot(implementation, timer, A, a_d, b_d, result_d);
    }
    if(implementation.SymGS_implemented){
        bench_SymGS(implementation, timer, A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, x_d, y_d);
    }
    // bench_SPMV(implementation, timer, A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, x_d, y_d);
    // bench_Dot(implementation, timer, A, x_d, y_d, result_d);
    // bench_SymGS(implementation, timer, A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, x_d, y_d);
    // other functions to be benchmarked
}

