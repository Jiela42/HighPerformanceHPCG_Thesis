#include "hip/hip_runtime.h"
#include "HPCG_versions/striped_warp_reduction_hipified.cuh"
#include "UtilLib/cuda_utils_hipified.hpp"
#include <iostream>

__global__ void reduce_sums(DataType * array, local_int_t num_elements, DataType * result_d){

    __shared__ DataType intermediate_sums[warpSize];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    DataType my_sum = 0.0;

    for (local_int_t i = tid; i < num_elements; i += blockDim.x * gridDim.x){
        my_sum += array[i];
    }

    
    for (int offset = warpSize/2; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xffffffffffffffffULL, my_sum, offset);
    }
    
    __syncthreads();
    
    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }
    
    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (threadIdx.x < num_warps) {
        my_sum = intermediate_sums[threadIdx.x];
    } else {
        my_sum = 0.0;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xffffffffffffffffULL, my_sum, offset);
    }

    __syncthreads();

    if(threadIdx.x == 0){
        result_d[blockIdx.x] = my_sum;
    }
}

__global__ void striped_warp_reduction_dot_kernel(
    local_int_t num_rows,
    DataType * x_d,
    DataType * y_d,
    DataType * result_d
){

    __shared__ DataType intermediate_sums[warpSize];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    // warp_id within the block
    int warp_id = threadIdx.x / warpSize;

    // first we reduce as much as we can without cooperation
    DataType my_sum = 0.0;

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        my_sum += x_d[i] * y_d[i];
    }

    // now we cooperatively reduce the sum
    for (int offset = warpSize/2; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xffffffffffffffffULL, my_sum, offset);
    }

    __syncthreads();

    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    
    if (threadIdx.x < num_warps) {
        my_sum = intermediate_sums[threadIdx.x];
    } else {
        my_sum = 0.0;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xffffffffffffffffULL, my_sum, offset);
    }
    
    __syncthreads();

    if(threadIdx.x == 0){
        result_d[blockIdx.x] = my_sum;
    }
}

/*
uses MPI_Allreduce to sum up across all ranks
*/
template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeDot(
    striped_Matrix<T>& A, //we only pass A for the metadata
    T * x_d,
    T * y_d,
    T * result_d
    ){

    int coop_num = this->dot_cooperation_number;

    local_int_t num_rows = A.get_num_rows();
    int num_threads = 1024;
    int max_threads = NUM_PHYSICAL_CORES;
    int max_blocks = 4 * max_threads / num_threads + 1;
    int num_blocks = std::min((int)(num_rows/(num_threads*coop_num)), max_blocks);
    num_blocks = max(num_blocks, 1);

    bool seperate_reduction_needed = num_blocks > 1;

    // allocate memory for the intermediate vector if needed
    T *intermediate_sums_d;
    if (seperate_reduction_needed){
        CHECK_CUDA(hipMalloc(&intermediate_sums_d, num_blocks * sizeof(T)));
    } else{
        intermediate_sums_d = result_d;
    }

    striped_warp_reduction_dot_kernel<<<num_blocks, num_threads>>>(
        num_rows, x_d, y_d, intermediate_sums_d
    );

    int num_inter_results = num_blocks;
    CHECK_CUDA(hipDeviceSynchronize());

    while (num_inter_results > 1){
        int num_threads = 1024;
        num_blocks = std::min(num_inter_results/(num_threads*coop_num), max_blocks);
        num_blocks = max(num_blocks, 1);

        if(num_blocks == 1){
            reduce_sums<<<1, num_threads>>>(intermediate_sums_d, num_inter_results, result_d);
        } else {
            reduce_sums<<<num_blocks, num_threads>>>(intermediate_sums_d, num_inter_results, intermediate_sums_d);
        }

        CHECK_CUDA(hipDeviceSynchronize());
        num_inter_results = num_blocks;
    }

    if(seperate_reduction_needed){
        CHECK_CUDA(hipFree(intermediate_sums_d));
    }
}

template class striped_warp_reduction_Implementation<DataType>;