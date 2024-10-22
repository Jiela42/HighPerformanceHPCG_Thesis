// just hello world to check if cmake setup works
#include <iostream>
#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/banded_Matrix.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveBanded.cuh"
#include "cuda_utils.hpp"

int main() {

    int num = 4;

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


    std::cout << "sanity_check read and write of matrix" << std::endl;

    std::string str_nx = std::to_string(A.get_nx());
    std::string str_ny = std::to_string(A.get_ny());
    std::string str_nz = std::to_string(A.get_nz());

    A.write_to_file();
    sparse_CSR_Matrix<double> A_from_file;
    A_from_file.read_from_file(str_nx, str_ny, str_nz, "cpp");

    A.compare_to(A_from_file);

    std::cout << "sanity check complete" << std::endl;

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


    std::cout << "cuSparse SPMV done" << std::endl;

    // Run a banded SPMV

    // get a banded version of A
    banded_Matrix<double> banded_A;
    banded_A.banded_3D27P_Matrix_from_CSR(A);

    // allocate memory on the device

    int num_bands = banded_A.get_num_bands();


    double * banded_A_d;
    double * y_min_i_d;

    CHECK_CUDA(cudaMalloc(&banded_A_d, num_bands * num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_min_i_d, num_rows * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(banded_A_d, banded_A.get_values().data(), num_bands * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_min_i_d, banded_A.get_j_min_i().data(), num_rows * sizeof(int), cudaMemcpyHostToDevice));

    // call the SPMV
    naiveBanded_Implementation<double> naiveBanded;

    naiveBanded.compute_SPMV(
        A,
                          A_row_ptr_d, A_col_idx_d, A_values_d,
                          x_d, y_d
    );

    // naiveBanded.compute_SPMV(
    //     A,
    //     banded_A_d,
    //     num_rows, num_cols,
    //     num_bands,
    //     banded_A.get_j_min_i().data(),
    //     x_d, y_d);

    // copy the result back into a new vector
    std::vector<double> y_banded(num_rows, 0.0);
    CHECK_CUDA(cudaMemcpy(y_banded.data(), y_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));


    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));

    std::cout << "Naive Banded SPMV done" << std::endl;
    // banded_A.print();



    // // compare the results
    // for (int i = 0; i < num_rows; i++) {
    //     if (y[i] != y_banded[i]) {
    //         std::cerr << "Error: cuSparse and Naive Banded SPMV results do not match." << std::endl;
    //         return 1;
    //     }
    // }

    return 0;
}