// wv[i] = alpha * xv[i] + beta * yv[i]
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"

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


__global__ void scalar_vector_multi_GPU_kernel(
    local_int_t num_rows,
    DataType alpha,
    DataType * x_d,
    DataType * w_d,
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
){

    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(row, nx, ny, nz, dimx, dimy);
        w_d[hi] = alpha * x_d[hi];
    }
}


__global__ void waxpb1y_multi_GPU_kernel(
    local_int_t num_rows,
    DataType alpha,
    DataType * x_d,
    DataType * y_d,
    DataType * w_d,
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(row, nx, ny, nz, dimx, dimy);
        w_d[hi] = alpha * x_d[hi] + y_d[hi];
    }
}

__global__ void w1xpb1y_multi_GPU_kernel(
    local_int_t num_rows,
    DataType * x_d,
    DataType * y_d,
    DataType * w_d,
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
){

    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(row, nx, ny, nz, dimx, dimy);
        w_d[hi] = x_d[hi] + y_d[hi];
    }
}

__global__ void waxpby_multi_GPU_kernel(
    local_int_t num_rows,
    DataType alpha,
    DataType * x_d,
    DataType beta,
    DataType * y_d,
    DataType * w_d,
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
){

    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        local_int_t hi = local_i_to_halo_i(row, nx, ny, nz, dimx, dimy);
        w_d[hi] = alpha * x_d[hi] + beta * y_d[hi];
    }
}


template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeWAXPBY(
    Halo *x_d,
    Halo *y_d,
    Halo *w_d,
    T alpha, T beta,
    Problem *problem,
    bool updateHalo
    ){

    assert(x_d->dimx == y_d->dimx && x_d->dimx == w_d->dimx);
    assert(x_d->dimy == y_d->dimy && x_d->dimy == w_d->dimy);
    assert(x_d->dimz == y_d->dimz && x_d->dimz == w_d->dimz);
    assert(x_d->nx == y_d->nx && x_d->nx == w_d->nx);
    assert(x_d->ny == y_d->ny && x_d->ny == w_d->ny);
    assert(x_d->nz == y_d->nz && x_d->nz == w_d->nz);

    local_int_t dimx = x_d->dimx;
    local_int_t dimy = x_d->dimy;
    local_int_t dimz = x_d->dimz;
    local_int_t nx = x_d->nx;
    local_int_t ny = x_d->ny;
    local_int_t nz = x_d->nz;
    
    local_int_t num_rows = problem->nx * problem->ny * problem->nz;
    int num_threads = 1024;
    int num_blocks = std::min(MAX_NUM_BLOCKS, (int) ceiling_division(num_rows, num_threads));

    if(alpha == 0.0 && beta == 0.0){
        CHECK_CUDA(cudaMemset(w_d->x_d, 0, dimx * dimy * dimz * sizeof(T)));
    }

    else if(alpha == 0.0){
        scalar_vector_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, beta, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(beta == 0.0){
        scalar_vector_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(alpha == 1.0 and beta == 1.0){
        w1xpb1y_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, x_d->x_d, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(alpha == 1.0){
        waxpb1y_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, beta, y_d->x_d, x_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(beta == 1.0){
        waxpb1y_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d->x_d, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else{
        waxpby_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d->x_d, beta, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    if(updateHalo){
        this->ExchangeHalo(w_d, problem);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

}

template class striped_multi_GPU_Implementation<DataType>;