#include <iostream>
#include <iomanip>
#include "HPCG_versions/cusparse.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/generations.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"

#define COMPARE_NORMS 1


void print_rrNorm(int nx, int ny, int nz){

    // make CSR matrix
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x = generate_random_vector(A.get_num_cols(), 42);


    // copy A, x, y to device
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, x.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, y.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // run the SymGS
    cuSparse_Implementation<double> cuSparse;
    cuSparse.compute_SymGS(A,
                          x_d, y_d);

    // calculate the l2 norm on the device
    double random_x_norm = relative_residual_norm_for_SymGS(
                                            A,
                                            x_d, y_d
                                        );

    double zero_x_norm = -1.0;

    if (COMPARE_NORMS){

        CHECK_CUDA(cudaMemset(x_d, 0, x.size() * sizeof(double)));

        cuSparse.compute_SymGS(A,
                            x_d, y_d);

        // calculate the l2 norm on the device
        zero_x_norm = relative_residual_norm_for_SymGS(
                                            A,
                                            x_d, y_d
                                            );
    }


    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    std::cout << "The relative residual norm for " << nx << "x" << ny << "x" << nz << " for random initialization is " << random_x_norm << std::endl;
    
    if(COMPARE_NORMS){
        std::cout << "The relative residual norm for " << nx << "x" << ny << "x" << nz << " for zero initialization is " << zero_x_norm << std::endl;
        std::cout << "The difference is " << zero_x_norm - random_x_norm << std::endl;
    }
}

int main(){

    std::cout << "Getting the relative residual norm" << std::endl;
    print_rrNorm(2, 2, 2);
    print_rrNorm(4, 4, 4);
    print_rrNorm(8, 8, 8);
    print_rrNorm(16, 16, 16);
    print_rrNorm(24, 24, 24);
    print_rrNorm(32, 32, 32);
    print_rrNorm(64, 64, 64);
    print_rrNorm(128, 64, 64);
    print_rrNorm(128, 128, 64);
    print_rrNorm(128, 128, 128);
    print_rrNorm(256, 128, 128);
}