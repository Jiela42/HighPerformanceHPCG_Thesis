#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>
#include <mpi.h>

__inline__ __device__ global_int_t local_i_to_halo_i(
    int i, 
    int nx, int ny, int nz,
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

__global__ void reduce_sums_multi_GPU(double * array, int num_elements, double * result_d){

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

    if(threadIdx.x == 0){
        result_d[blockIdx.x] = my_sum;
    }
}

__global__ void striped_warp_reduction_multi_GPU_dot_kernel(
    int num_rows,
    double * x_d,
    double * y_d,
    double * result_d,
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
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
        local_int_t hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
        my_sum += x_d[hi] * y_d[hi];
        // printf("i = %d, x_d[i] = %f, y_d[i] = %f\n", i, x_d[i], y_d[i]);
    }

    // now we cooperatively reduce the sum
    // if(tid == 0){
    //     printf("before warp level reduction\n");
    // }

    for (int offset = 16; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    // if(tid == 0){
    //     printf("after warp level reduction\n");
    // }
    __syncthreads();

    if (lane == 0){
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // if (tid == 0){
    //     printf("write to shared memory done\n");
    // }

    // now we reduce the intermediate sums
    if (threadIdx.x < 32){
        my_sum = intermediate_sums[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
    }

    // if(tid == 0){
    //     printf("Reduced stuff in shared memory\n");
    // }

    __syncthreads();

    // printf("my_sum = %f\n", my_sum);

    // if (tid == 0){
    //     printf("let's write to global memory\n");
    // }
    if(threadIdx.x == 0){
        // printf("yo imma write to global memory\n");
        // printf("blockIdx.x = %d\n", blockIdx.x);
        // printf("resuld_d: %p\n", result_d);

        if (my_sum != 0.0){

        // printf("result_d[%d] = %f\n", blockIdx.x, result_d[blockIdx.x]);
        }
        result_d[blockIdx.x] = my_sum;
    }

    // if(tid == 0){
    //     printf("wrote to result_d\n");
    // }

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
    // std::cout << "Running dot product with striped warp reduction" << std::endl;
    // we compute z = xy

    int num_rows = x_d->nx * x_d->ny * x_d->nz;
    int num_threads = 1024;
    int max_threads = NUM_PHYSICAL_CORES;
    int max_blocks = 4 * max_threads / num_threads + 1;
    int num_blocks = std::min(num_rows/(num_threads*coop_num), max_blocks);
    // we need at least one block
    num_blocks = max(num_blocks, 1);

    bool seperate_reduction_needed = num_blocks > 1;

    // allocate memory for the intermediate vector if needed
    double *intermediate_sums_d;
    if (seperate_reduction_needed){
        // std::cout << "seperate reduction needed" << std::endl;
        // std::cout << "num_blocks = " << num_blocks << std::endl;
        CHECK_CUDA(cudaMalloc(&intermediate_sums_d, num_blocks * sizeof(double)));
    } else{
        intermediate_sums_d = result_d;
        // std::cout << "no seperate reduction needed we write to result_d directly" << std::endl;
    }

    // std::cout << "num_rows = " << num_rows << std::endl;

    striped_warp_reduction_multi_GPU_dot_kernel<<<num_blocks, num_threads>>>(
        num_rows, x_d->x_d, y_d->x_d, intermediate_sums_d, x_d->nx, x_d->ny, x_d->nz, x_d->dimx, x_d->dimy
    );

    int num_inter_results = num_blocks;
    CHECK_CUDA(cudaDeviceSynchronize());

    // std::cout << "num_inter_results = " << num_inter_results << std::endl;

    while (num_inter_results > 1){
        // std::cout << "we enter the loop" << std::endl;
        // std::cout << "num_inter_results = " << num_inter_results << std::endl;
        int num_threads = 1024;
        num_blocks = std::min(num_inter_results/(num_threads*coop_num), max_blocks);
        // we need at least one block
        num_blocks = max(num_blocks, 1);

        if(num_blocks == 1){
            reduce_sums_multi_GPU<<<1, num_threads>>>(intermediate_sums_d, num_inter_results, result_d);
        } else {
            reduce_sums_multi_GPU<<<num_blocks, num_threads>>>(intermediate_sums_d, num_inter_results, intermediate_sums_d);
        }


        CHECK_CUDA(cudaDeviceSynchronize());
        num_inter_results = num_blocks;
    }

    DataType my_result;
    CHECK_CUDA(cudaMemcpy(&my_result, result_d, sizeof(DataType), cudaMemcpyDeviceToHost));
    DataType result_h;
    MPI_Allreduce(&my_result, &result_h, 1, MPIDataType, MPI_SUM, MPI_COMM_WORLD);
    CHECK_CUDA(cudaMemcpy(result_d, &result_h, sizeof(DataType), cudaMemcpyHostToDevice));

    // std::cout<< "after the loop"<< std::endl;
    // use a kernel to reduce the intermediate sums
    // reduce_sums<<<1, num_threads>>>(intermediate_sums_d, num_blocks, result_d);

    // CHECK_CUDA(cudaDeviceSynchronize());

    if(seperate_reduction_needed){
        // std::cout << "freeing intermediate_sums_d" << std::endl;
        CHECK_CUDA(cudaFree(intermediate_sums_d));
    }

    // std::cout << "done with dot product" << std::endl;

}

template class striped_multi_GPU_Implementation<DataType>;