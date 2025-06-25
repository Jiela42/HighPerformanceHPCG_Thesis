// This file implements WAXPBY operations (w = alpha*x + beta*y) using CUDA for multiple GPUs,
// with specialized kernels for performance-optimized corner cases.

#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"

/**
 * @brief Convert local index to halo index in the extended domain.
 *
 * @param i     Local index within the interior.
 * @param nx, ny, nz Local dimensions of the interior.
 * @param dimx, dimy Dimensions of the extended domain including halos.
 * @return global_int_t Halo index corresponding to local index i.
 */
__inline__ __device__ global_int_t local_i_to_halo_i(
    local_int_t i, 
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t dimx, local_int_t dimy
    )
{
    return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

/**
 * @brief Kernel for scalar-vector multiplication: w = alpha * x
 *
 * Multiplies each element of vector x by scalar alpha and stores in w.
 */
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

/**
 * @brief Kernel for w = alpha * x + y
 *
 * Computes a weighted sum of x and y with alpha scaling x.
 */
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

/**
 * @brief Kernel for w = x + y
 *
 * Adds two vectors element-wise.
 */
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

/**
 * @brief Kernel for w = alpha * x + beta * y
 *
 * General weighted sum of two vectors.
 */
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

/**
 * @brief Computes w = alpha*x + beta*y using optimized CUDA kernels for halo-extended vectors.
 *
 * Selects specialized kernels for common scalar cases to improve performance.
 *
 * @param x_d, y_d  Input vectors.
 * @param w_d       Output vector.
 * @param alpha, beta Scalars for combination.
 * @param problem   Geometry and layout metadata.
 * @param updateHalo Whether to perform halo exchange on w_d after computation.
 */
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

    // Handle special cases with zero scalars for efficiency
    if(alpha == 0.0 && beta == 0.0){
        CHECK_CUDA(cudaMemset(w_d->x_d, 0, dimx * dimy * dimz * sizeof(T)));
    }
    else if(alpha == 0.0){
        // w = beta * y
        scalar_vector_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, beta, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(beta == 0.0){
        // w = alpha * x
        scalar_vector_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(alpha == 1.0 and beta == 1.0){
        // w = x + y
        w1xpb1y_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, x_d->x_d, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(alpha == 1.0){
        // w = y + beta * x (note swapped arguments in kernel call)
        waxpb1y_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, beta, y_d->x_d, x_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else if(beta == 1.0){
        // w = alpha * x + y
        waxpb1y_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d->x_d, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    else{
        // General case
        waxpby_multi_GPU_kernel<<<num_blocks, num_threads>>>(num_rows, alpha, x_d->x_d, beta, y_d->x_d, w_d->x_d, nx, ny, nz, dimx, dimy);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    if(updateHalo){
        this->ExchangeHalo(w_d, problem);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

}

template class striped_multi_GPU_Implementation<DataType>;