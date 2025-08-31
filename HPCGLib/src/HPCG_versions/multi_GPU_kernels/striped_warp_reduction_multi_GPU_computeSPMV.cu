// This file implements sparse matrix-vector multiplication (SPMV) using striped matrix storage
// and warp-level reductions on multiple GPUs with halo-based data access.

#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define PIPELINE_DEPTH 8
#define NUM_THREADS_PER_BLOCK 256

namespace cg = cooperative_groups;

/**
 * @brief Converts a local coordinate index to the corresponding index within a halo data region.
 * @param i Local linear index.
 * @param nx, ny, nz Local grid dimensions.
 * @param dimx, dimy Dimensions of the data region including halos.
 * @return Index into the halo data region.
 */
__inline__ __device__ global_int_t local_i_to_halo_i(
    int i,
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
)
{
    return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

__constant__ local_int_t j_min_i_d[27];

__inline__ __device__ global_int_t local_i_to_global_i(
    local_int_t i, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0
    )
    {
        /*
        int local_i_x = i % nx;
        int local_i_y = (i % (nx * ny)) / nx;
        int local_i_z = i / (nx * ny);
        return gi0 + local_i_x + local_i_y * gnx + local_i_z * (gnx * gny);*/
        return gi0 + (i % nx) + (((i / nx) % ny) * gnx) + ((i / (nx * ny)) * (gnx * gny)); //should be equivalent to the above
}

__inline__ __device__ local_int_t global_i_to_halo_i(
    global_int_t i,
    local_int_t nx, local_int_t ny, local_int_t nz,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
    )
    {
        return ((i % gnx) - px * nx + 1) +
            ((((i / gnx) % gny) - py * ny + 1) * (nx + 2)) +
            (((i / (gnx * gny)) - pz * nz + 1) * ((nx + 2) * (ny + 2))); //should be equivalent to the above
}

__global__ void striped_warp_reduction_multi_GPU_SPMV_kernel_old(
        DataType* striped_A,
        local_int_t num_rows, int num_stripes, local_int_t * j_min_i,
        double* x, double* y, local_int_t nx, local_int_t ny, local_int_t nz, 
        global_int_t gnx, global_int_t gny, global_int_t gnz, 
        global_int_t gi0,
        int px, int py, int pz
    )
{
    // printf("striped_warp_reduction_SPMV_kernel\n");
    local_int_t cooperation_number = 4;
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t lane = threadIdx.x % cooperation_number;

    // every thread computes one or more rows of the matrix
    for (local_int_t i = tid/cooperation_number; i < num_rows; i += (blockDim.x * gridDim.x)/cooperation_number) {
        // compute the matrix-vector product for the ith row
        // convert i to global index
        global_int_t gi = local_i_to_global_i(i, nx, ny, nz, gnx, gny, gnz, gi0);
        double sum_i = 0;
        for (local_int_t stripe = lane; stripe < num_stripes; stripe += cooperation_number) {
            local_int_t gj = gi + j_min_i[stripe]; //use the global index gi to find global index gj
            if (gj >= 0 && gj < gnx * gny * gnz) {
                //convert gj to halo coordinate hj which is the memory location of gj in the halo struct
                local_int_t hj =  global_i_to_halo_i(gj, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
                local_int_t current_row = i * num_stripes;
                if(hj>=0 && hj<(nx+2)*(ny+2)*(nz+2))sum_i += striped_A[current_row + stripe] * x[hj];
            }
        }
        // now let's reduce the sum_i to a single value using warp-level reduction
        for(int offset = cooperation_number/2; offset > 0; offset /= 2){
            sum_i += __shfl_down_sync(0xFFFFFFFF, sum_i, offset);
        }
        
        __syncthreads();
        
        if (lane == 0){
            //convert gi to halo coordinate hi which is the memory location of gi in the halo struct
            local_int_t hi =  global_i_to_halo_i(gi, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
            if(hi>=0 && hi<(nx+2)*(ny+2)*(nz+2))y[hi] = sum_i;
        }
    }
}

/**
 * @brief Kernel for sparse matrix-vector multiplication using striped storage format.
 *
 * Each thread cooperatively computes a matrix row using warp-level reduction, accessing halo-aware vectors.
 *
 * @param striped_A   Striped matrix values.
 * @param num_rows    Number of rows in the matrix.
 * @param num_stripes Number of stripes per row.
 * @param j_min_i     Stripe offsets for each row.
 * @param x, y        Input and output vectors (with halo).
 * @param nx, ny, nz  Local grid dimensions.
 * @param gnx, gny, gnz Global grid dimensions.
 * @param gi0         Global index offset.
 * @param px, py, pz  Process grid coordinates.
 */
__global__ void striped_warp_reduction_multi_GPU_SPMV_kernel(
        DataType* striped_A,
        local_int_t num_rows, int num_stripes,
        double* x, double* y, local_int_t nx, local_int_t ny, local_int_t nz, 
        local_int_t dimx, local_int_t dimy, local_int_t dimz
    )
{
    local_int_t cooperation_number = 4;
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t lane = threadIdx.x % cooperation_number;

    // every thread computes one or more rows of the matrix
    for (local_int_t i = tid / cooperation_number; i < num_rows; i += (blockDim.x * gridDim.x) / cooperation_number) {
        // compute the matrix-vector product for the ith row
        // convert i to global index
        local_int_t hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
        double sum_i = 0;
        for (local_int_t stripe = lane; stripe < num_stripes; stripe += cooperation_number) {
            local_int_t hj = hi + j_min_i_d[stripe];
            local_int_t current_row = i * num_stripes;
            sum_i += striped_A[current_row + stripe] * x[hj];
        }
        // now let's reduce the sum_i to a single value using warp-level reduction
        for (int offset = cooperation_number / 2; offset > 0; offset /= 2) {
            sum_i += __shfl_down_sync(0xFFFFFFFF, sum_i, offset);
        }

        __syncthreads();

        if (lane == 0) {
            // convert gi to halo coordinate hi which is the memory location of gi in the halo struct
            if (hi >= 0 && hi < (dimx) * (dimy) * (dimz)) y[hi] = sum_i;
        }
    }
}

//clumn-major
__global__ void columnMajor_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID

    //gate
    if(tid >= num_rows) return;
    
    local_int_t hi = local_i_to_halo_i(tid, nx, ny, nz, dimx, dimy);

    DataType sum_i = 0;
    for(int stripe = 0; stripe < num_stripes; stripe++){
        local_int_t coeff_i = tid + stripe * num_rows;
        DataType coeff = A_d[coeff_i];
        int v_i = hi + j_min_i_d[stripe];
        sum_i += coeff * x_d[v_i];
    }

    y[hi] = sum_i;
}

__global__ void blocked_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    int global_warp_id = tid / warpSize; // global warp ID
    int lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    
    //gate
    if(tid >= num_rows) return;
    
    local_int_t hi = local_i_to_halo_i(tid, nx, ny, nz, dimx, dimy);

    DataType sum_i = 0;
    A_d += warpSize * num_stripes * global_warp_id; // move to the start of the current warp's rows
    for(int stripe = 0; stripe < num_stripes; stripe++){
        DataType coeff = A_d[lane_id];
        int v_i = hi + j_min_i_d[stripe];
        sum_i += coeff * x_d[v_i];
        A_d += warpSize; // move to the next row
    }

    y[hi] = sum_i;
}

__global__ void partial_blocked_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz,
    int border_width_x, int border_width_y, int border_width_z, // thickness of the faces (not) to be computed
    bool compute_border,
    bool compute_inner
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    int global_warp_id = tid / warpSize; // global warp ID
    int lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    
    //gate
    if(tid >= num_rows) return;

    int i_x = tid % nx; // x-coordinate in the local grid
    int i_y = (tid / nx) % ny; // y-coordinate in the local grid
    int i_z = tid / (nx * ny); // z-coordinate in the local grid

    if(!compute_border && (
        i_x < border_width_x || i_x >= nx - border_width_x ||
        i_y < border_width_y || i_y >= ny - border_width_y ||
        i_z < border_width_z || i_z >= nz - border_width_z
    )) return; // skip if we are not computing the border and the current index is in the border region

    if(!compute_inner && (
        i_x >= border_width_x && i_x < nx - border_width_x &&
        i_y >= border_width_y && i_y < ny - border_width_y &&
        i_z >= border_width_z && i_z < nz - border_width_z
    )) return; // skip if we are not computing the inner region and the current index is in the inner region
    
    local_int_t hi = local_i_to_halo_i(tid, nx, ny, nz, dimx, dimy);

    DataType sum_i = 0;
    A_d += warpSize * num_stripes * global_warp_id; // move to the start of the current warp's rows
    for(int stripe = 0; stripe < num_stripes; stripe++){
        DataType coeff = A_d[lane_id];
        int v_i = hi + j_min_i_d[stripe];
        sum_i += coeff * x_d[v_i];
        A_d += warpSize; // move to the next row
    }

    y[hi] = sum_i;
}

__global__ void border_blocked_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz,
    int border_width_x, int border_width_y, int border_width_z, // thickness of the faces (not) to be computed
    local_int_t *border_idx_d, // thread to index mapping for border computation
    local_int_t num_border_values // number of border values to compute
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    
    //gate
    if(tid >= num_border_values) return;
    
    //figure out which element of border to compute
    local_int_t my_val = border_idx_d[tid]; // get the index for this thread
    
    local_int_t hi = local_i_to_halo_i(my_val, nx, ny, nz, dimx, dimy);

    local_int_t base = (my_val / warpSize) * warpSize * num_stripes; // base index for the current
    local_int_t stride = warpSize;
    local_int_t offset = my_val % warpSize;

    DataType sum_i = 0;
    for(int stripe = 0; stripe < num_stripes; stripe++){
        DataType coeff = A_d[base + stripe * stride + offset];
        int v_i = hi + j_min_i_d[stripe];
        sum_i += coeff * x_d[v_i];
    }

    y[hi] = sum_i;
}

__global__ void inner_blocked_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz,
    int border_width_x, int border_width_y, int border_width_z, // thickness of the faces (not) to be computed
    local_int_t num_inner_values // number of interior values to compute
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    
    //gate
    if(tid >= num_inner_values) return;
    
    //figure out which element of border to compute
    local_int_t inner_width_x = nx - 2 * border_width_x;
    local_int_t inner_width_y = ny - 2 * border_width_y;
    local_int_t inner_width_z = nz - 2 * border_width_z;
    int offset_x = border_width_x + tid % inner_width_x; // compute the offset in the x-direction
    int offset_y = border_width_y + (tid % (inner_width_x * inner_width_y)) / inner_width_x; // compute the offset in the y-direction
    int offset_z = border_width_z + (tid / (inner_width_x * inner_width_y)); // compute the offset in the z-direction
    local_int_t my_val = offset_x + offset_y * nx + offset_z * nx * ny; // compute the local index in the inner region
    
    local_int_t hi = local_i_to_halo_i(my_val, nx, ny, nz, dimx, dimy);

    local_int_t base = (my_val / warpSize) * warpSize * num_stripes; // base index for the current
    local_int_t stride = warpSize;
    local_int_t offset = my_val % warpSize;

    DataType sum_i = 0;
    for(int stripe = 0; stripe < num_stripes; stripe++){
        DataType coeff = A_d[base + stripe * stride + offset];
        int v_i = hi + j_min_i_d[stripe];
        sum_i += coeff * x_d[v_i];
    }

    y[hi] = sum_i;
}

__global__ void two_elems_multi_GPU_SPMV_kernel(
    DataType* __restrict__ A_d,
    local_int_t num_rows, int num_stripes,
    double* __restrict__ x_d, double* __restrict__ y, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    int global_warp_id = tid / warpSize; // global warp ID
    int lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    int my_val = tid * 2; // each thread computes two elements

    //gate
    if(my_val >= num_rows) return;
    
    local_int_t hi = local_i_to_halo_i(my_val, nx, ny, nz, dimx, dimy);

    DataType sum1 = 0;
    DataType sum2 = 0;
    A_d += warpSize * 2 * num_stripes * global_warp_id; // move to the start of the current warp's rows
    for(int stripe = 0; stripe < num_stripes; stripe++){
        double2* data = reinterpret_cast<double2*>(A_d); // reinterpret A_d to double2 for two elements
        double2 coeffs = data[lane_id]; // load two coefficients at once
        double coeff1 = coeffs.x;
        double coeff2 = coeffs.y;
        int v1 = hi + j_min_i_d[stripe];
        int v2 = hi + 1 + j_min_i_d[stripe];
        sum1 += coeff1 * x_d[v1];
        sum2 += coeff2 * x_d[v2];
        A_d += warpSize * 2; // move to the next row
    }

    // write both results to the output vector
    y[my_val] = sum1;
    y[my_val + 1] = sum2;
}

__global__ void pipelined_columnMajor_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    DataType* x_d, DataType* y_d, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz,
    int num_elems_to_compute
)
{
    // Shared memory for DataType buffering
    __shared__ alignas(16) DataType sA_buffer[PIPELINE_DEPTH][NUM_THREADS_PER_BLOCK]; // Shared memory for matrix values
    __shared__ alignas(16) DataType sx_buffer[PIPELINE_DEPTH][NUM_THREADS_PER_BLOCK]; // Shared memory for vector values

    //Create thread block group for cooperative operations
    auto block = cg::this_thread_block();

    // Pipeline object for managing async operations - CHANGED TO THREAD SCOPE
    auto pipeline = cuda::make_pipeline();

    int totalThreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;

    for(int elem = 0; elem < num_elems_to_compute; elem ++){
        const local_int_t block_start_A = elem * totalThreads + blockIdx.x * NUM_THREADS_PER_BLOCK; // Start index for this block in the matrix
        const local_int_t block_start_x = local_i_to_halo_i(block_start_A, nx, ny, nz, dimx, dimy); // Start index for this block in the 3d padded halo vector
        const local_int_t elements_this_block = min(static_cast<local_int_t>(NUM_THREADS_PER_BLOCK), num_rows - block_start_A);
        const local_int_t copy_size = elements_this_block * sizeof(DataType);
        const local_int_t my_val = block_start_x + threadIdx.x; // index which this thread will compute

        // Only process if we have valid elements
        if (block_start_A + threadIdx.x >= num_rows) return;

        int producer_idx = 0;
        int consumer_idx = 0;

        //prefill the buffer
        for(int tile_idx = 0; tile_idx < PIPELINE_DEPTH - 1; tile_idx++){
            local_int_t tile_start_A = block_start_A + tile_idx * num_rows;
            local_int_t tile_start_x = block_start_x + j_min_i_d[tile_idx];

            pipeline.producer_acquire();

            cuda::memcpy_async(block,
                sA_buffer[producer_idx], // Shared memory buffer
                A_d + tile_start_A, // Global memory source
                copy_size, // Size of the copy
                pipeline // Pipeline for async operation
            );
            cuda::memcpy_async(block, 
                sx_buffer[producer_idx], // Shared memory buffer
                x_d + tile_start_x, // Global memory source
                copy_size, // Size of the copy
                pipeline // Pipeline for async operation
            );

            producer_idx = (producer_idx + 1) % PIPELINE_DEPTH;
            pipeline.producer_commit();
        }

        // Process tiles with overlap
        DataType my_sum = 0;
        for(int tile_idx = 0; tile_idx < num_stripes; tile_idx++){

            const local_int_t next_load_tile = tile_idx + (PIPELINE_DEPTH - 1);
            if(next_load_tile < num_stripes){
                local_int_t next_tile_start_A = block_start_A + next_load_tile * num_rows;
                local_int_t next_tile_start_x = block_start_x + j_min_i_d[next_load_tile];
                // Issue async copy for the next tile
                pipeline.producer_acquire();
                cuda::memcpy_async(block, 
                    sA_buffer[producer_idx], // Shared memory buffer
                    A_d + next_tile_start_A, // Global memory source
                    copy_size, // Size of the copy
                    pipeline // Pipeline for async operation
                );
                cuda::memcpy_async(block, 
                    sx_buffer[producer_idx], // Shared memory buffer
                    x_d + next_tile_start_x, // Global memory source
                    copy_size, // Size of the copy
                    pipeline // Pipeline for async operation
                );
                producer_idx = (producer_idx + 1) % PIPELINE_DEPTH;
                pipeline.producer_commit();
            }

            // Wait for current tile and compute
            pipeline.consumer_wait();
            // __syncthreads() since using thread-scope pipeline with shared memory
            __syncthreads();
            my_sum += sA_buffer[consumer_idx][threadIdx.x] * sx_buffer[consumer_idx][threadIdx.x];
            consumer_idx = (consumer_idx + 1) % PIPELINE_DEPTH;
            pipeline.consumer_release();

        }
        y_d[my_val] = my_sum;
    }
}

__global__ void color_wise_multi_GPU_SPMV_kernel(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y_d, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz,
    int bx, int by, int bz,
    int px, int py, int pz,
    int block_size
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    int color = tid / block_size; // color index
    int lane = tid % block_size; // lane index within the color

    // calculate number of values for that color
    local_int_t num_color_cols = nx / bx;
    local_int_t num_color_rows = ny / by;
    local_int_t num_color_faces = nz / bz;

    // How is the vector colored
    local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
    local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color

    local_int_t color_offs_x_local = (color_offs_x_global + (bx - ((px * nx) % bx)) % bx) % bx;
    local_int_t color_offs_y_local = (color_offs_y_global + (by - ((py * ny) % by)) % by) % by;
    local_int_t color_offs_z_local = (color_offs_z_global + (bz - ((pz * nz) % bz)) % bz) % bz;

    num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

    //gate
    if(color >= bx * by * bz) return; // if color is out of bounds
    if(lane >= num_nodes_with_color) return;

    // Find the (ix,iy,iz) position of the node for this color.
    local_int_t ix = lane % num_color_cols;
    local_int_t iy = ((lane % (num_color_cols * num_color_rows))) / num_color_cols;
    local_int_t iz = lane / (num_color_cols * num_color_rows);

    // Map to full local grid coordinates.
    ix = ix * bx + color_offs_x_local;
    iy = iy * by + color_offs_y_local;
    iz = iz * bz + color_offs_z_local;

    // Compute local and halo indices.
    local_int_t li = ix + iy * nx + iz * nx * ny;
    local_int_t hi = local_i_to_halo_i(li, nx, ny, nz, dimx, dimy);

    DataType sum_i = 0;
    local_int_t base = color * block_size * num_stripes;
    for(int stripe = 0; stripe < num_stripes; stripe++){
        DataType coeff = A_d[base + stripe * block_size + lane];
        local_int_t v_i = hi + j_min_i_d[stripe];
        sum_i += coeff * x_d[v_i];
    }

    y_d[hi] = sum_i;
}

__global__ void color_wise_multi_GPU_SPMV_kernel_opt(
    DataType* A_d,
    local_int_t num_rows, int num_stripes,
    double* x_d, double* y_d, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    local_int_t dimx, local_int_t dimy, local_int_t dimz,
    int bx, int by, int bz,
    int px, int py, int pz,
    int block_size
)
{
    //get IDs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    int num_colors = bx * by * bz; // total number of colors

    for(int color = 0; color < num_colors; color++){
        // calculate number of values for that color
        local_int_t num_color_cols = nx / bx;
        local_int_t num_color_rows = ny / by;
        local_int_t num_color_faces = nz / bz;
    
        // How is the vector colored
        local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
        local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
        local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color
    
        local_int_t color_offs_x_local = (color_offs_x_global + (bx - ((px * nx) % bx)) % bx) % bx;
        local_int_t color_offs_y_local = (color_offs_y_global + (by - ((py * ny) % by)) % by) % by;
        local_int_t color_offs_z_local = (color_offs_z_global + (bz - ((pz * nz) % bz)) % bz) % bz;
    
        num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
        num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
        num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;
    
        int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;
    
        //gate
        if(tid >= num_nodes_with_color) continue;;
    
        // Find the (ix,iy,iz) position of the node for this color.
        local_int_t ix = tid % num_color_cols;
        local_int_t iy = ((tid % (num_color_cols * num_color_rows))) / num_color_cols;
        local_int_t iz = tid / (num_color_cols * num_color_rows);
    
        // Map to full local grid coordinates.
        ix = ix * bx + color_offs_x_local;
        iy = iy * by + color_offs_y_local;
        iz = iz * bz + color_offs_z_local;
    
        // Compute local and halo indices.
        local_int_t li = ix + iy * nx + iz * nx * ny;
        local_int_t hi = local_i_to_halo_i(li, nx, ny, nz, dimx, dimy);
    
        DataType sum_i = 0;
        local_int_t base = color * block_size * num_stripes;
        for(int stripe = 0; stripe < num_stripes; stripe++){
            DataType coeff = A_d[base + stripe * block_size + tid];
            local_int_t v_i = hi + j_min_i_d[stripe];
            sum_i += coeff * x_d[v_i];
        }
    
        y_d[hi] = sum_i;
    }
}

void compute_COO_SPMV(
    striped_partial_Matrix<DataType>& A,
    Halo *x_d, Halo *y_d, // the vectors x and y are already on the device
    Problem *problem
) {
    global_int_t nnz_COO = A.nnz_COO;
    global_int_t *row = A.row_COO;
    global_int_t *col = A.col_COO;
    int *col_to_rank = A.col_to_rank_COO;
    DataType *data = A.data_COO;
    DataType *result = (DataType *) malloc(nnz_COO * sizeof(DataType));
    int count_per_rank[problem->size];
    for(int i = 0; i < nnz_COO; i++) {
    }
        
        
}


/**
 * @brief Launches the striped SPMV kernel on multiple GPUs with halo-aware inputs.
 *
 * @param A        Matrix with striped format.
 * @param x_d      Input vector (device, halo-aware).
 * @param y_d      Output vector (device, halo-aware).
 * @param problem  Geometry and decomposition metadata.
 */
template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeSPMV(
        striped_partial_Matrix<T>& A,
        Halo *x_d, Halo *y_d, // the vectors x and y are already on the device
        Problem *problem,
        bool exchangeHalo,
        bool exchangeHaloOverlap
    ) {
        local_int_t num_rows = A.get_num_rows(); // exclude halo rows
        int num_stripes = A.get_num_stripes();
        T * striped_A_d = A.get_values_d();

        // since every thread is working on one or more rows we need to base the number of threads on that
        int elems_per_thread = 1;
        int num_threads = NUM_THREADS_PER_BLOCK;
        int total_threads_needed = (num_rows + elems_per_thread - 1) / elems_per_thread;
        int num_blocks = (total_threads_needed + num_threads - 1) / num_threads;

        //move j_min_i to constant memory
        std::vector<local_int_t> j_min_i_h(27);
        CHECK_CUDA(cudaMemcpy(j_min_i_h.data(), A.get_j_min_i_halo_d(), 27 * sizeof(local_int_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyToSymbol(j_min_i_d, j_min_i_h.data(), 27 * sizeof(local_int_t)));

        striped_warp_reduction_multi_GPU_SPMV_kernel_old<<<num_blocks, num_threads>>>(
            striped_A_d, num_rows, num_stripes, A.get_j_min_i_d(), x_d->x_d, y_d->x_d, problem->nx, problem->ny, problem->nz,
            problem->gnx, problem->gny, problem->gnz, problem->gi0, problem->px, problem->py, problem->pz
        );

        striped_warp_reduction_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, 
            num_rows, num_stripes, 
            x_d->x_d, y_d->x_d,
            problem->nx, problem->ny, problem->nz,
            x_d->dimx, x_d->dimy, x_d->dimz
        );

        columnMajor_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, 
            num_rows, num_stripes, 
            x_d->x_d, y_d->x_d,
            problem->nx, problem->ny, problem->nz,
            x_d->dimx, x_d->dimy, x_d->dimz
        );

        blocked_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, 
            num_rows, num_stripes, 
            x_d->x_d, y_d->x_d,
            problem->nx, problem->ny, problem->nz,
            x_d->dimx, x_d->dimy, x_d->dimz
        );

        pipelined_columnMajor_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, 
            num_rows, num_stripes, 
            (DataType*)x_d->x_d, (DataType*)y_d->x_d,
            problem->nx, problem->ny, problem->nz,
            x_d->dimx, x_d->dimy, x_d->dimz,
            (num_rows + num_threads - 1) / num_threads
        );

        color_wise_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, 
            num_rows, num_stripes, 
            x_d->x_d, y_d->x_d,
            problem->nx, problem->ny, problem->nz,
            x_d->dimx, x_d->dimy, x_d->dimz,
            A.bx, A.by, A.bz,
            problem->px, problem->py, problem->pz,
            A.block_size
        );

        //version for inner/outer computation overlap with halo exchange
        
        // if(!exchangeHalo){ //do not exchange halo
        //     blocked_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
        //         striped_A_d, 
        //         num_rows, num_stripes, 
        //         x_d->x_d, y_d->x_d,
        //         problem->nx, problem->ny, problem->nz,
        //         x_d->dimx, x_d->dimy, x_d->dimz
        //     );
        // } else if (!exchangeHaloOverlap){ //exchange halo but not overlap
        //     CHECK_CUDA(cudaDeviceSynchronize()); //wait for previous computation to finish
        //     blocked_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
        //         striped_A_d, 
        //         num_rows, num_stripes, 
        //         x_d->x_d, y_d->x_d,
        //         problem->nx, problem->ny, problem->nz,
        //         x_d->dimx, x_d->dimy, x_d->dimz
        //     );
        //     CHECK_CUDA(cudaDeviceSynchronize());
        //     this->ExchangeHalo(y_d, problem);
        // }else{ // exchange halo with computation-communication overlap
        //     int num_blocks_border = (A.get_num_border_values() + num_threads - 1) / num_threads; // number of blocks for border computation
        //     cudaStreamSynchronize(*(y_d->streams[26])); // Wait for previous inner computation to finish
        //     cudaStreamSynchronize(*(y_d->streams[27])); // Wait for previous border computation to finish
        //     border_blocked_multi_GPU_SPMV_kernel<<<num_blocks_border, num_threads, 0, *(y_d->streams[26])>>>( // launch on stream 26 the border computation
        //         striped_A_d, 
        //         num_rows, num_stripes, 
        //         x_d->x_d, y_d->x_d,
        //         problem->nx, problem->ny, problem->nz,
        //         x_d->dimx, x_d->dimy, x_d->dimz,
        //         1, 1, 1,
        //         A.get_border_idx_d(), // thread to index mapping for border computation
        //         A.get_num_border_values() // number of border values to compute
        //     );
        //     int num_blocks_inner = (A.get_num_inner_values() + num_threads - 1) / num_threads; // number of blocks for inner computation
        //     inner_blocked_multi_GPU_SPMV_kernel<<<num_blocks_inner, num_threads, 0, *(y_d->streams[27])>>>( // launch on stream 27 the inner computation
        //         striped_A_d, 
        //         num_rows, num_stripes, 
        //         x_d->x_d, y_d->x_d,
        //         problem->nx, problem->ny, problem->nz,
        //         x_d->dimx, x_d->dimy, x_d->dimz,
        //         1, 1, 1,
        //         A.get_num_inner_values() // number of border values to compute
        //     );
        //     this->ExchangeHalo(y_d, problem);
        // }

        // synchronize the device
        CHECK_CUDA(cudaDeviceSynchronize());
    }

// explicit template instantiation
template class striped_multi_GPU_Implementation<DataType>;