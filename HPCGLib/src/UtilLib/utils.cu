#include "UtilLib/utils.cuh"

#include "UtilLib/cuda_utils.hpp"
#include "HPCG_versions/naiveStriped.cuh" // we need this for the matrix vector multiplication kernel
#include "HPCG_versions/striped_warp_reduction.cuh" // we need this for the matrix vector multiplication kernel

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

__global__ void diff_and_norm_kernel(int num_rows, double *Ax, double *y, double *y_reduced) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    __shared__ double intermediate_sums[32];

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        double d = Ax[row] - y[row];
        my_sum += d * d;
    }
    // reduce along warps
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    __syncthreads();

    if (lane == 0) {
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // sync threads in the block
    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    if(threadIdx.x == 0){
        y_reduced[blockIdx.x] = my_sum;
        // printf("y_reduced[%d]: %f\n", blockIdx.x, y_reduced[blockIdx.x]);
    }
}

__global__ void square_fusedReduction_kernel(int num_rows, double *y, double *y_reduced) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    __shared__ double intermediate_sums[32];

    // printf("Hello from the kernel\n");

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        my_sum += y[row] * y[row];
    }

    // reduce along warps
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    __syncthreads();

    if (lane == 0) {
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // sync threads in the block
    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    if(threadIdx.x == 0){
        y_reduced[blockIdx.x] = my_sum;
        // printf("y_reduced[%d]: %f\n", blockIdx.x, y_reduced[blockIdx.x]);
    }
}

__global__ void compute_restriction_kernel(
    int num_rows,
    double * Axf,
    double * rf,
    double * rc,
    int * f2c_operator
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        rc[row] = rf[f2c_operator[row]] - Axf[f2c_operator[row]];
    }
}

__global__ void compute_prolongation_kernel(
    int num_rows,
    double * xc,
    double * xf,
    int * f2c_operator
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {

        // if(f2c_operator[row] < 0){

        //     printf("f2c_operator is the problem for row %d\n", row);
        //     printf("f2c_operator[%d]: %d\n", row, f2c_operator[row]);
        //     printf("xc is the problem for row %d\n", row);
        //     printf("xc[row]: %f\n", xc[row]);
        //     printf("xf is the problem for row %d\n", row);
        //     printf("xf[f2c_operator[%d]]: %f\n",row, xf[f2c_operator[row]]);
        // }


        xf[f2c_operator[row]] += xc[row];
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

    // std::cout << "Hello from L2_norm_for_Device_Vector" << std::endl;

    
    // Compute the squared vector and start reducing
    int blockSize = 1024;
    int numBlocks = (num_rows + blockSize - 1) / (blockSize*8);
    // we need to make sure there is at least one block
    numBlocks = std::max(numBlocks, 1);
    CHECK_CUDA(cudaMalloc(&y_squared, numBlocks * sizeof(double)));
    square_fusedReduction_kernel<<<numBlocks, blockSize, 0, stream>>>(num_rows, y, y_squared);

    // Use Thrust to compute the sum of the squared vector
    thrust::device_ptr<double> y_squared_ptr(y_squared);
    *result = std::sqrt(thrust::reduce(thrust::cuda::par.on(stream), y_squared_ptr, y_squared_ptr + numBlocks));

    // Free device memory
    CHECK_CUDA(cudaFreeAsync(y_squared, stream));

}

double L2_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
){

    int num_rows = A.get_num_rows();
    int * row_ptr = A.get_row_ptr_d();
    int * col_idx = A.get_col_idx_d();
    double * values = A.get_values_d();

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
    double L2_norm = std::sqrt(thrust::reduce(diff_ptr, diff_ptr + numBlocks));

    // Free device memory
    CHECK_CUDA(cudaFree(Ax));
    CHECK_CUDA(cudaFree(diff));

    return L2_norm;
}

double L2_norm_for_SymGS(
    striped_Matrix<double> & A,
    double * x,
    double * y
){

    striped_warp_reduction_Implementation<double> implementation;
    int num_rows = A.get_num_rows();
    
    // Allocate memory for Ax on the device
    double *Ax;
    double *result;
    CHECK_CUDA(cudaMalloc(&Ax, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result, 1 * sizeof(double)));

    CHECK_CUDA(cudaMemset(Ax, 0, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMemset(result, 0, 1 * sizeof(double)));

    // Perform matrix-vector multiplication: Ax = A * x

    implementation.compute_SPMV(A, x, Ax);
    implementation.compute_WAXPBY(A, Ax, y, Ax, 1.0, -1.0);
    implementation.compute_Dot(A, Ax, Ax, result);

    // copy result over
    double result_h;
    CHECK_CUDA(cudaMemcpy(&result_h, result, 1 * sizeof(double), cudaMemcpyDeviceToHost));


    // Free device memory
    CHECK_CUDA(cudaFree(Ax));
    CHECK_CUDA(cudaFree(result));

    return std::sqrt(result_h);
}


double relative_residual_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
){

    int num_rows = A.get_num_rows();
    int * row_ptr = A.get_row_ptr_d();
    int * col_idx = A.get_col_idx_d();
    double * values = A.get_values_d();

    
    // make new cuda stream for the vector y computation
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    double L2_norm_y;

    L2_norm_for_Device_Vector(stream, num_rows, y, &L2_norm_y);
    // the order of the calls is important, because L2_norm_for_SymGS will synchronize the default stream and not return until it does
    double L2_norm = L2_norm_for_SymGS(A, x, y);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return L2_norm / L2_norm_y;
}

double relative_residual_norm_for_SymGS(
    striped_Matrix<double> & A,
    double * x,
    double * y
){

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    int num_stripes = A.get_num_stripes();
    int * j_min_i = A.get_j_min_i_d();
    double * striped_A_d = A.get_values_d();

    // std::cout << "relative_residual_norm_for_SymGS" << std::endl;
    // make new cuda stream for the vector y computation
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    double L2_norm_y;

    L2_norm_for_Device_Vector(stream, num_rows, y, &L2_norm_y);
    // the order of the calls is important, because L2_norm_for_SymGS will synchronize the default stream and not return until it does
    double L2_norm = L2_norm_for_SymGS(A, x, y);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaDeviceSynchronize());
    // std::cout << "L2_norm: " << L2_norm << std::endl;
    // std::cout << "L2_norm_y: " << L2_norm_y << std::endl;
    return L2_norm / L2_norm_y;
}









