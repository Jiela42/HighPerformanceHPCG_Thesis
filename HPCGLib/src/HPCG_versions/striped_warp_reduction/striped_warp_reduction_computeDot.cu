


#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

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

    __syncthreads();

    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = 16; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    __syncthreads();

    if(tid == 0){
        *result_d = my_sum;
    }
}

__global__ void striped_warp_reduction_dot_kernel(
    int num_rows,
    double * x_d,
    double * y_d,
    double * result_d
){

    __shared__ double intermediate_sums[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    // warp_id within the block
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

    __syncthreads();

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

    __syncthreads();

    // printf("my_sum = %f\n", my_sum);
    if(threadIdx.x == 0){
        result_d[blockIdx.x] = my_sum;
        // if (my_sum != 0.0){

        // printf("result_d[%d] = %f\n", blockIdx.x, result_d[blockIdx.x]);
        // }
    }
}

template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeDot(
    striped_Matrix<T>& A, //we only pass A for the metadata
    T * x_d,
    T * y_d,
    T * result_d
    ){
    
    int coop_num = this->dot_cooperation_number;
    // we compute z = xy

    int num_rows = A.get_num_rows();
    int num_threads = 1024;
    int num_blocks = std::min(num_rows/(num_threads*coop_num), MAX_NUM_BLOCKS);
    // we need at least one block
    num_blocks = max(num_blocks, 1);

    // allocate memory for the intermediate vector
    double *intermediate_sums_d;

    CHECK_CUDA(cudaMalloc(&intermediate_sums_d, num_blocks * sizeof(double)));

    // std::cout << "calling the kernel with " << num_blocks << " blocks" << std::endl;

    striped_warp_reduction_dot_kernel<<<num_blocks, num_threads>>>(
        num_rows, x_d, y_d, intermediate_sums_d
    );

    // CHECK_CUDA(cudaDeviceSynchronize());

    // reduce_sums<<<1, num_threads>>>(intermediate_sums_d, num_blocks, result_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    // use thrust to reduce the intermediate sums
    // thrust::device_ptr<double> thrust_intermediate_sums(intermediate_sums_d);
    // double result = thrust::reduce(thrust_intermediate_sums, thrust_intermediate_sums + num_blocks, 0.0, thrust::plus<double>());


    // print intermediate sums
    // double * intermediate_sums = new double[num_blocks];
    // CHECK_CUDA(cudaMemcpy(intermediate_sums, intermediate_sums_d, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < num_blocks; i++){
    //     std::cout << "intermediate_sums[" << i << "] = " << intermediate_sums[i] << std::endl;
    // }

    // // write the result to the device
    // CHECK_CUDA(cudaMemcpy(result_d, &result, sizeof(double), cudaMemcpyHostToDevice));

    // use a kernel to reduce the intermediate sums
    reduce_sums<<<1, num_threads>>>(intermediate_sums_d, num_blocks, result_d);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(intermediate_sums_d));

}

template class striped_warp_reduction_Implementation<double>;