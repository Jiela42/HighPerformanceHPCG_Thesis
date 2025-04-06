#include "hip/hip_runtime.h"
#include "HPCG_versions/striped_multi_GPU_hipified.cuh"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"
#include <iostream>
#include <mpi.h>



__inline__ __device__ global_int_t local_i_to_halo_i(
    local_int_t i,
    int nx, int ny, int nz,
    int dimx, int dimy
    )
    {
        return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

__global__ void reduce_sums_multi_GPU(DataType * array, local_int_t num_elements, DataType * result_d){

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

__global__ void striped_warp_reduction_multi_GPU_dot_kernel(
    local_int_t num_rows,
    DataType * x_d,
    DataType * y_d,
    DataType * result_d,
    int nx, int ny, int nz,
    int dimx, int dimy
){

    __shared__ DataType intermediate_sums[warpSize];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    DataType my_sum = 0.0;

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
        my_sum += x_d[hi] * y_d[hi];
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

template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeDot(
    Halo * x_d,
    Halo * y_d,
    T * result_d
    ){

    assert(x_d->dimx == y_d->dimx);
    assert(x_d->dimy == y_d->dimy);
    assert(x_d->dimz == y_d->dimz);
    assert(x_d->nx == y_d->nx);
    assert(x_d->ny == y_d->ny);
    assert(x_d->nz == y_d->nz);


    int coop_num = this->dot_cooperation_number;

    local_int_t num_rows = x_d->nx * x_d->ny * x_d->nz;
    int num_threads = 1024;
    int max_threads = NUM_PHYSICAL_CORES;
    int max_blocks = 4 * max_threads / num_threads + 1;
    int num_blocks = std::min((int) (num_rows/(num_threads*coop_num)), max_blocks);
    num_blocks = max(num_blocks, 1);

    bool seperate_reduction_needed = num_blocks > 1;

    DataType *intermediate_sums_d;
    if (seperate_reduction_needed){
        CHECK_CUDA(hipMalloc(&intermediate_sums_d, num_blocks * sizeof(DataType)));
        CHECK_CUDA(hipMemset(intermediate_sums_d,0 ,num_blocks * sizeof(DataType)));
    } else{
        intermediate_sums_d = result_d;
    }

    striped_warp_reduction_multi_GPU_dot_kernel<<<num_blocks, num_threads>>>(
        num_rows, x_d->x_d, y_d->x_d, intermediate_sums_d, x_d->nx, x_d->ny, x_d->nz, x_d->dimx, x_d->dimy
    );

    int num_inter_results = num_blocks;
    CHECK_CUDA(hipDeviceSynchronize());

    while (num_inter_results > 1){
        int num_threads = 1024;
        num_blocks = std::min((int)num_inter_results/(num_threads*coop_num), max_blocks);
        num_blocks = max(num_blocks, 1);

        if(num_blocks == 1){
            reduce_sums_multi_GPU<<<1, num_threads>>>(intermediate_sums_d, num_inter_results, result_d);
        } else {
            reduce_sums_multi_GPU<<<num_blocks, num_threads>>>(intermediate_sums_d, num_inter_results, intermediate_sums_d);
        }

        CHECK_CUDA(hipDeviceSynchronize());
        num_inter_results = num_blocks;
    }

    DataType my_result;
    CHECK_CUDA(hipMemcpy(&my_result, result_d, sizeof(DataType), hipMemcpyDeviceToHost));
    DataType result_h;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&my_result, &result_h, 1, MPIDataType, MPI_SUM, MPI_COMM_WORLD);
    CHECK_CUDA(hipMemcpy(result_d, &result_h, sizeof(DataType), hipMemcpyHostToDevice));

    if(seperate_reduction_needed){
        CHECK_CUDA(hipFree(intermediate_sums_d));
    }

}

template class striped_multi_GPU_Implementation<DataType>;