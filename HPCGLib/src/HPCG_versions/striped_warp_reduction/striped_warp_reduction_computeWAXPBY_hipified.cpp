#include "hip/hip_runtime.h"
// wv[i] = alpha * xv[i] + beta * yv[i]
#include "HPCG_versions/striped_warp_reduction_hipified.cuh"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/utils_hipified.cuh"

__global__ void scalar_vector_mult_kernel(
    local_int_t num_rows,
    double alpha,
    double * x_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = alpha * x_d[row];
    }
}


__global__ void waxpb1y_kernel(
    local_int_t num_rows,
    double alpha,
    double * x_d,
    double * y_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = alpha * x_d[row] + y_d[row];
    }
}

__global__ void w1xpb1y_kernel(
    local_int_t num_rows,
    double * x_d,
    double * y_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = x_d[row] + y_d[row];
    }
}

__global__ void waxpby_kernel(
    local_int_t num_rows,
    double alpha,
    double * x_d,
    double beta,
    double * y_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = alpha * x_d[row] + beta * y_d[row];
    }
}

template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeWAXPBY(
    striped_Matrix<T>& A, //we only pass A for the metadata
    T * x_d,
    T * y_d,
    T * w_d,
    T alpha, T beta
    ){

    local_int_t num_rows = A.get_num_rows();
    int num_threads = 1024;
    int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, num_threads));

    if(alpha == 0.0 && beta == 0.0){
        CHECK_CUDA(hipMemset(w_d, 0, num_rows * sizeof(T)));
    }

    else if(alpha == 0.0){
        scalar_vector_mult_kernel<<<num_blocks, num_threads>>>(num_rows, beta, y_d, w_d);
    }
    else if(beta == 0.0){
        scalar_vector_mult_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d, w_d);
    }
    else if(alpha == 1.0 and beta == 1.0){
        w1xpb1y_kernel<<<num_blocks, num_threads>>>(num_rows, x_d, y_d, w_d);
    }
    else if(alpha == 1.0){
        waxpb1y_kernel<<<num_blocks, num_threads>>>(num_rows, beta, y_d, x_d, w_d);
    }
    else if(beta == 1.0){
        waxpb1y_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d, y_d, w_d);
    }
    else{
        waxpby_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d, beta, y_d, w_d);
    }

    CHECK_CUDA(hipDeviceSynchronize());

}

template class striped_warp_reduction_Implementation<double>;