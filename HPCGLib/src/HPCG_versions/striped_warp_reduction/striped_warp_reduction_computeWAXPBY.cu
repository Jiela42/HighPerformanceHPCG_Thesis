// wv[i] = alpha * xv[i] + beta * yv[i]
#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"

__global__ void scalar_vector_mult_kernel(
    int num_rows,
    double alpha,
    double * x_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = alpha * x_d[row];
    }
}


__global__ void waxpb1y_kernel(
    int num_rows,
    double alpha,
    double * x_d,
    double * y_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = alpha * x_d[row] + y_d[row];
    }
}

__global__ void w1xpb1y_kernel(
    int num_rows,
    double * x_d,
    double * y_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        w_d[row] = x_d[row] + y_d[row];
    }
}

__global__ void waxpby_kernel(
    int num_rows,
    double alpha,
    double * x_d,
    double beta,
    double * y_d,
    double * w_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x){
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
    
    int num_rows = A.get_num_rows();
    int num_threads = 1024;
    int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, num_threads));

    if(alpha == 0.0 && beta == 0.0){
        CHECK_CUDA(cudaMemset(w_d, 0, num_rows * sizeof(T)));
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

    CHECK_CUDA(cudaDeviceSynchronize());

}

template class striped_warp_reduction_Implementation<double>;