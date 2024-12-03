


#include "HPCG_versions/banded_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>

__global__ void reduce_sums(double * array, int num_elements, double * result_d){

    __shared__ double intermediate_sums[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    double my_sum = 0.0;

    for (int i = tid; i < num_elements; i += blockDim.x * gridDim.x){
        my_sum += array[i];
    }

    for (int offset = 16; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = 16; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    if(tid == 0){
        *result_d = my_sum;
    }
}

__global__ void banded_warp_reduction_dot_kernel(
    int num_rows,
    double * x_d,
    double * y_d,
    double * result_d
){

    __shared__ double intermediate_sums[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // first we reduce as much as we can without cooperation
    double my_sum = 0.0;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        // if (y_d[i] != 0.0){
        //     printf("y_d[%d] = %f\n", i, y_d[i]);
        // }
        my_sum += x_d[i] * y_d[i];
        // printf("i = %d, x_d[i] = %f, y_d[i] = %f\n", i, x_d[i], y_d[i]);
    }

    // now we cooperatively reduce the sum

    for (int offset = 16; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // now we reduce the intermediate sums
    if (threadIdx.x < 32){
        my_sum = intermediate_sums[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    if(tid == 0){
        result_d[blockIdx.x] = my_sum;
        // printf("result_d = %f\n", *result_d);
    }
}

template <typename T>
void banded_warp_reduction_Implementation<T>::banded_warp_reduction_computeDot(
    banded_Matrix<T>& A, //we only pass A for the metadata
    T * x_d,
    T * y_d,
    T * result_d
    ){
    // we compute z = xy

    int num_rows = A.get_num_rows();
    int num_threads = 1024;
    int num_blocks = std::min(num_rows/8, 8*num_threads);
    num_blocks = 1;

    // get some shared memory for the subsequent reduction
    // double * intermediate_sums_d;
    // CHECK_CUDA(cudaMalloc(&intermediate_sums_d, num_blocks * sizeof(double)));    

    banded_warp_reduction_dot_kernel<<<num_blocks, num_threads>>>(
        num_rows, x_d, y_d, result_d
    );

    // CHECK_CUDA(cudaDeviceSynchronize());

    // reduce_sums<<<1, num_threads>>>(intermediate_sums_d, num_blocks, result_d);

    CHECK_CUDA(cudaDeviceSynchronize());



}

template class banded_warp_reduction_Implementation<double>;