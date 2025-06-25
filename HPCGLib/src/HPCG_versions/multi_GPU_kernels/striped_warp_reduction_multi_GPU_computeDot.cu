// This file implements a warp-reduction-based dot product on multiple GPUs using CUDA and MPI.
// Each block computes partial sums which are recursively reduced, followed by a global MPI_Allreduce.

#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include <iostream>
#include <mpi.h>
#include <chrono>
#include <thread>

__inline__ __device__ global_int_t local_i_to_halo_i(
    local_int_t i, 
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
    )
    {
        /*
        int local_i_x = i % nx;
        int local_i_y = (i % (nx * ny)) / nx;
        int local_i_z = i / (nx * ny);
        return (dimx * dimy) + dimx + 1 + local_i_x + local_i_y * dimx + local_i_z * (dimx * dimy);*/
        return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

/**
 * @brief Kernel to perform reduction of partial sums across threads.
 *
 * Each thread sums elements of the input array in a strided manner.
 * Warp-level shuffle operations reduce sums within warps.
 * Partial warp sums are stored in shared memory and further reduced by the first warp.
 * Final block-level sum is stored in result_d[blockIdx.x].
 */
__global__ void reduce_sums_multi_GPU(DataType * array, local_int_t num_elements, DataType * result_d){

    __shared__ DataType intermediate_sums[32];
    
    // Calculate global thread ID
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;      // lane index within the warp
    int warp_id = threadIdx.x / 32;   // warp index within the block

    DataType my_sum = 0.0;

    // Each thread sums multiple elements strided by total threads
    for (local_int_t i = tid; i < num_elements; i += blockDim.x * gridDim.x){
        my_sum += array[i];
    }

    // Warp-level reduction using shuffle down
    for (int offset = 16; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    __syncthreads();

    // First thread in each warp writes its partial sum to shared memory
    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // First warp reduces the warp sums stored in shared memory
    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = 16; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    __syncthreads();

    // Thread 0 writes the final block-level sum to the output array
    if(threadIdx.x == 0){
        result_d[blockIdx.x] = my_sum;
    }
}

/**
 * @brief Kernel to compute partial dot products using striped warp reduction.
 *
 * Each thread computes partial dot products of elements mapped from local to halo indexing.
 * Warp-level shuffle reductions reduce partial sums within warps.
 * Intermediate warp sums are stored and further reduced to block-level sums.
 * Final block sums are stored in result_d.
 */
__global__ void striped_warp_reduction_multi_GPU_dot_kernel(
    local_int_t num_rows,
    DataType * x_d,
    DataType * y_d,
    DataType * result_d,
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
){

    __shared__ DataType intermediate_sums[32];

    // Calculate global thread ID
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;      // lane index within the warp
    int warp_id = threadIdx.x / 32;   // warp index within the block

    // Each thread computes a partial sum of dot product elements
    DataType my_sum = 0.0;

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
        my_sum += x_d[hi] * y_d[hi];
    }

    // Warp-level reduction using shuffle down
    for (int offset = 16; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    __syncthreads();

    // First thread in each warp stores partial sum in shared memory
    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // First warp reduces the warp sums stored in shared memory
    if (threadIdx.x < 32){
        my_sum = intermediate_sums[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    __syncthreads();

    // Thread 0 writes the final block-level sum to the output array
    if(threadIdx.x == 0){
        if (my_sum != 0.0){

        }
        result_d[blockIdx.x] = my_sum;
    }

}

/**
 * @brief Computes the dot product between two halo-extended vectors on multiple GPUs.
 *
 * The operation uses warp-level reductions within each block, recursive block-level reductions,
 * and a final MPI_Allreduce to obtain the global result. The result is written to a device pointer.
 *
 * @param x_d  Input vector x (with halo).
 * @param y_d  Input vector y (with halo).
 * @param result_d  Output device pointer to store final dot product result.
 */
template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeDot(
    Halo * x_d,
    Halo * y_d,
    T * result_d
    ){

    // Ensure input dimensions match
    assert(x_d->dimx == y_d->dimx);
    assert(x_d->dimy == y_d->dimy);
    assert(x_d->dimz == y_d->dimz);
    assert(x_d->nx == y_d->nx);
    assert(x_d->ny == y_d->ny);
    assert(x_d->nz == y_d->nz);
    
    int coop_num = this->dot_cooperation_number;

    // Compute total number of rows (elements) in the local vector
    local_int_t num_rows = x_d->nx * x_d->ny * x_d->nz;

    // Set number of threads per block and calculate maximum blocks based on physical cores
    int num_threads = 1024;
    int max_threads = NUM_PHYSICAL_CORES;
    int max_blocks = 4 * max_threads / num_threads + 1;

    // Calculate number of blocks needed, considering cooperation number
    int num_blocks = std::min((int) (num_rows/(num_threads*coop_num)), max_blocks);
    // Ensure at least one block is launched
    num_blocks = max(num_blocks, 1);

    // Determine if a separate reduction step is needed based on number of blocks
    bool seperate_reduction_needed = num_blocks > 1;

    // Allocate memory for intermediate sums if needed
    DataType *intermediate_sums_d;
    if (seperate_reduction_needed){
        CHECK_CUDA(cudaMalloc(&intermediate_sums_d, num_blocks * sizeof(DataType)));
        CHECK_CUDA(cudaMemset(intermediate_sums_d,0 ,num_blocks * sizeof(DataType)));
    } else{
        intermediate_sums_d = result_d;
    }

    // Launch kernel to compute partial dot products and reduce within blocks
    striped_warp_reduction_multi_GPU_dot_kernel<<<num_blocks, num_threads>>>(
        num_rows, x_d->x_d, y_d->x_d, intermediate_sums_d, x_d->nx, x_d->ny, x_d->nz, x_d->dimx, x_d->dimy
    );

    int num_inter_results = num_blocks;
    CHECK_CUDA(cudaDeviceSynchronize());

    // Recursively reduce intermediate sums until only one remains
    while (num_inter_results > 1){
        int num_threads = 1024;
        num_blocks = std::min((int)num_inter_results/(num_threads*coop_num), max_blocks);
        // Ensure at least one block is launched
        num_blocks = max(num_blocks, 1);

        if(num_blocks == 1){
            // Final reduction step writes directly to result_d
            reduce_sums_multi_GPU<<<1, num_threads>>>(intermediate_sums_d, num_inter_results, result_d);
        } else {
            // Intermediate reduction steps overwrite intermediate_sums_d
            reduce_sums_multi_GPU<<<num_blocks, num_threads>>>(intermediate_sums_d, num_inter_results, intermediate_sums_d);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        num_inter_results = num_blocks;
    }

    // Copy partial result from device to host
    DataType my_result;
    CHECK_CUDA(cudaMemcpy(&my_result, result_d, sizeof(DataType), cudaMemcpyDeviceToHost));

    // Perform global MPI_Allreduce to sum partial results across all GPUs
    DataType result_h;
    MPI_Allreduce(&my_result, &result_h, 1, MPIDataType, MPI_SUM, MPI_COMM_WORLD);

    // Copy final result back to device memory
    CHECK_CUDA(cudaMemcpy(result_d, &result_h, sizeof(DataType), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());
}

template class striped_multi_GPU_Implementation<DataType>;