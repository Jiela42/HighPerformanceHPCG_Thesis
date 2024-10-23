#include "testing.hpp"

#include "cuda_utils.hpp"

void run_naiveBanded_tests(int nx, int ny, int nz){

    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // create the baseline and the UUT
    cuSparse_Implementation<double> cuSparse;
    naiveBanded_Implementation<double> naiveBanded;
    
    // create the matrix and vectors
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> x(nx*ny*nz, 1.0);

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    const int * A_row_ptr_data = A.get_row_ptr().data();
    const int * A_col_idx_data = A.get_col_idx().data();
    const double * A_values_data = A.get_values().data();

    int * A_row_ptr_d;
    int * A_col_idx_d;
    double * A_values_d;
    double * x_d;

    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));


    // test the SPMV function
    test_SPMV(cuSparse, naiveBanded, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d);

    // anything that got allocated also needs to be de-allocted
    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));
    CHECK_CUDA(cudaFree(x_d));
   
}