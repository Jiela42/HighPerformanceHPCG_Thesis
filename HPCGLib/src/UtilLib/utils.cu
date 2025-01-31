#include "UtilLib/utils.cuh"

#include "UtilLib/cuda_utils.hpp"
#include "HPCG_versions/naiveStriped.cuh" // we need this for the matrix vector multiplication kernel

#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

int ceiling_division(int numerator, int denominator) {
    return static_cast<int>(std::ceil(static_cast<double>(numerator) / denominator));
}

int next_smaller_power_of_two(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power / 2;
}

__global__ void matvec_mult_kernel(int num_rows, int *row_ptr, int *col_idx, double *values, double *x, double *Ax) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        Ax[row] = sum;
    }
}

__global__ void diff_and_norm_kernel(int num_rows, double *Ax, double *y, double *diff) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        double d = Ax[row] - y[row];
        diff[row] = d * d;
    }
}

__global__ void square_vector_kernel(int num_rows, double *y, double *y_squared) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        y_squared[row] = y[row] * y[row];
    }
}

void L2_norm_for_Device_Vector(
    cudaStream_t stream,
    int num_rows,
    double * y,
    double * result
){

    // Allocate memory for the squared vector on the device
    double *y_squared;

    CHECK_CUDA(cudaMalloc(&y_squared, num_rows * sizeof(double)));

    // Compute the squared vector
    int blockSize = 1024;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;
    square_vector_kernel<<<numBlocks, blockSize, 0, stream>>>(num_rows, y, y_squared);

    // Use Thrust to compute the sum of the squared vector
    thrust::device_ptr<double> y_squared_ptr(y_squared);
    *result = std::sqrt(thrust::reduce(thrust::cuda::par.on(stream), y_squared_ptr, y_squared_ptr + num_rows));

    // Free device memory
    CHECK_CUDA(cudaFreeAsync(y_squared, stream));

}

double L2_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int * row_ptr,
    int * col_idx,
    double * values,
    double * x,
    double * y
){
    // Allocate memory for Ax on the device
    double *Ax;
    double *diff;
    CHECK_CUDA(cudaMalloc(&Ax, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&diff, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemset(Ax, 0, num_rows * sizeof(double)));

    // Perform matrix-vector multiplication: Ax = A * x

    int blockSize = 1024;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;
    matvec_mult_kernel<<<numBlocks, blockSize>>>(num_rows, row_ptr, col_idx, values, x, Ax);
    CHECK_CUDA(cudaDeviceSynchronize());
    // // print Ax
    // double *Ax_h = (double *)malloc(num_rows * sizeof(double));
    // CHECK_CUDA(cudaMemcpy(Ax_h, Ax, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_rows; i++) {
    //     std::cout << Ax_h[i] << " ";
    // }
    // std::cout << std::endl;

    // Compute the difference and the squared differences
    diff_and_norm_kernel<<<numBlocks, blockSize>>>(num_rows, Ax, y, diff);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Use Thrust to compute the sum of the squared differences
    thrust::device_ptr<double> diff_ptr(diff);
    double L2_norm = std::sqrt(thrust::reduce(diff_ptr, diff_ptr + num_rows));

    // Free device memory
    CHECK_CUDA(cudaFree(Ax));
    CHECK_CUDA(cudaFree(diff));

    return L2_norm;
}

double L2_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int num_stripes,
    int * j_min_i,
    double * A,
    double * x,
    double * y
){    
    // Allocate memory for Ax on the device
    double *Ax;
    double *diff;
    CHECK_CUDA(cudaMalloc(&Ax, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&diff, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemset(Ax, 0, num_rows * sizeof(double)));

    // Perform matrix-vector multiplication: Ax = A * x

    int blockSize = 1024;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;
    
    naiveStriped_SPMV_kernel<<<numBlocks, blockSize>>>(
        A,
        num_rows, num_stripes, j_min_i,
        x, Ax);

    
    CHECK_CUDA(cudaDeviceSynchronize());

    // // print Ax
    // double *Ax_h = (double *)malloc(num_rows * sizeof(double));
    // CHECK_CUDA(cudaMemcpy(Ax_h, Ax, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_rows; i++) {
    //     std::cout << Ax_h[i] << " ";
    // }
    // std::cout << std::endl;


    // Compute the difference and the squared differences
    diff_and_norm_kernel<<<numBlocks, blockSize>>>(num_rows, Ax, y, diff);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Use Thrust to compute the sum of the squared differences
    thrust::device_ptr<double> diff_ptr(diff);
    double L2_norm = std::sqrt(thrust::reduce(diff_ptr, diff_ptr + num_rows));

    // Free device memory
    CHECK_CUDA(cudaFree(Ax));
    CHECK_CUDA(cudaFree(diff));

    return L2_norm;
}


double relative_residual_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int * row_ptr,
    int * col_idx,
    double * values,
    double * x,
    double * y
){

    
    // make new cuda stream for the vector y computation
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    double L2_norm_y;

    L2_norm_for_Device_Vector(stream, num_rows, y, &L2_norm_y);
    // the order of the calls is important, because L2_norm_for_SymGS will synchronize the default stream and not return until it does
    double L2_norm = L2_norm_for_SymGS(num_rows, num_cols, row_ptr, col_idx, values, x, y);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return L2_norm / L2_norm_y;
}

double relative_residual_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int num_stripes,
    int * j_min_i,
    double * A,
    double * x,
    double * y
){
    // std::cout << "relative_residual_norm_for_SymGS" << std::endl;
    // make new cuda stream for the vector y computation
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    double L2_norm_y;

    L2_norm_for_Device_Vector(stream, num_rows, y, &L2_norm_y);
    // the order of the calls is important, because L2_norm_for_SymGS will synchronize the default stream and not return until it does
    double L2_norm = L2_norm_for_SymGS(num_rows, num_cols, num_stripes, j_min_i, A, x, y);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaDeviceSynchronize());
    // std::cout << "L2_norm: " << L2_norm << std::endl;
    // std::cout << "L2_norm_y: " << L2_norm_y << std::endl;
    return L2_norm / L2_norm_y;
}









