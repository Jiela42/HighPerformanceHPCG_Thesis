// just hello world to check if cmake setup works
#include <iostream>
#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "cuda_utils.hpp"

int main() {

    int num = 4;

    std::cout << "Hello, World!" << std::endl;
    std::cout << "Starting Matrix Generation" << std::endl;
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(num, num, num);
    
    // call spmv
    std::vector<double> x(num*num*num, 1.0);
    std::vector<double> y(num*num*num, 0.0);

    cuSparse_Implementation<double> cuSparse;

    // we actually have to put shit on the gpu here.
    int * A_row_ptr_d;
    int * A_col_idx_d;
    double * A_values_d;
    double * x_d;
    double * y_d;

    sparse_CSR_Matrix<double> A = problem.first;

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    const int * A_row_ptr_data = A.get_row_ptr().data();
    const int * A_col_idx_data = A.get_col_idx().data();
    const double * A_values_data = A.get_values().data();

    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));


    cuSparse.compute_SPMV(A,
                          A_row_ptr_d, A_col_idx_d, A_values_d,
                          x_d, y_d);

    // and now we need to copy the result back and de-allocte the memory

    CHECK_CUDA(cudaMemcpy(y.data(), y_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    std::cout << "Matrix Generation Complete" << std::endl;
    return 0;
}