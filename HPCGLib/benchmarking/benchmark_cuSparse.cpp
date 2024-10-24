#include "benchmark.hpp"

void run_cuSparse_3d27p__benchmarks(int nx, int ny, int nz, std::string folder_path){
    // create the timer

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> x(nx*ny*nz, 0.7);

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    const int * A_row_ptr_data = A.get_row_ptr().data();
    const int * A_col_idx_data = A.get_col_idx().data();
    const double * A_values_data = A.get_values().data();
    
    CudaTimer timer(nx, ny, nz, nnz, "41-44", "3d_27pt", "cuSparse", folder_path);
    cuSparse_Implementation<double> implementation;

    // Allocate the memory on the device
    int * A_row_ptr_d;
    int * A_col_idx_d;
    double * A_values_d;
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(cuSparse_implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d);

    // free the memory
    cudaFree(A_row_ptr_d);
    cudaFree(A_col_idx_d);
    cudaFree(A_values_d);
    cudaFree(x_d);
    cudaFree(y_d);
}