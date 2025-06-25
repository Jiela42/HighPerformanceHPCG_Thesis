// This file defines a CUDA kernel and host function to perform a parallel Symmetric Gauss-Seidel (SymGS)
// iteration using striped box coloring for structured 3D problems on multiple GPUs.

#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"


/**
 * @brief Convert a local node index to its global index in the full 3D grid.
 *
 * @param i   Local node index.
 * @param nx, ny, nz  Local grid dimensions.
 * @param gnx, gny, gnz Global grid dimensions.
 * @param gi0  Global index offset for this local domain.
 * @return Corresponding global node index.
 */
__inline__ __device__ global_int_t local_i_to_global_i(
    local_int_t i, 
    local_int_t nx, local_int_t ny, local_int_t nz, 
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0
    )
{
    local_int_t local_i_x = i % nx;
    local_int_t local_i_y = (i % (nx * ny)) / nx;
    local_int_t local_i_z = i / (nx * ny);
    return gi0 + local_i_x + local_i_y * gnx + local_i_z * (gnx * gny);
}

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

/**
 * @brief Convert a global node index to the corresponding index in the local halo buffer.
 *
 * @param i   Global node index.
 * @param nx, ny, nz  Local grid dimensions.
 * @param gnx, gny, gnz Global grid dimensions.
 * @param gi0  Global index offset for this local domain.
 * @param px, py, pz  Process coordinates in the process grid.
 * @return Index into the local halo buffer.
 */
__inline__ __device__ local_int_t global_i_to_halo_i(
    local_int_t i,
    local_int_t nx, local_int_t ny, local_int_t nz,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    local_int_t px, local_int_t py, local_int_t pz
    )
{
    local_int_t global_j_x = i % gnx;
    local_int_t global_j_y = (i % (gnx * gny)) / gnx;
    local_int_t global_j_z = i / (gnx * gny);
    local_int_t halo_j_x = global_j_x - px * nx + 1;
    local_int_t halo_j_y = global_j_y - py * ny + 1;
    local_int_t halo_j_z = global_j_z - pz * nz + 1;
    return halo_j_x + halo_j_y * (nx+2) + halo_j_z * ((nx+2) * (ny+2));
}

/**
 * @brief CUDA kernel for one forward or backward step of Symmetric Gauss-Seidel (SymGS)
 *        for a single color using striped box coloring on a structured 3D grid.
 *
 * Each thread group cooperates to process one node of the current color, performing a striped
 * matrix-vector operation and a warp reduction to update the unknowns.
 *
 * @param cooperation_number  Number of threads cooperating per node.
 * @param color              Color index to process in this kernel call.
 * @param bx,by,bz           Box coloring factors in each dimension.
 * @param nx,ny,nz           Local grid dimensions.
 * @param num_rows,num_cols  Matrix dimensions.
 * @param num_stripes        Number of stripes per row.
 * @param diag_offset        Offset of diagonal in the stripe storage.
 * @param j_min_i            Array of minimum column indices per stripe.
 * @param striped_A          Matrix values in striped storage.
 * @param x                  Solution vector (updated in place).
 * @param y                  Right-hand side vector.
 * @param gnx,gny,gnz        Global grid dimensions.
 * @param gi0                Global index offset for this domain.
 * @param px,py,pz           Process grid coordinates.
 */
__global__ void striped_box_coloring_half_SymGS_kernel_old(
    int cooperation_number,
    int color, int bx, int by, int bz,
    local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t num_rows, local_int_t num_cols,
    int num_stripes, int diag_offset,
    DataType * striped_A,
    DataType * x, DataType * y,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
){
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t coop_group_id = tid / cooperation_number;
    local_int_t lane = tid % cooperation_number;
    local_int_t num_coop_groups = blockDim.x * gridDim.x / cooperation_number;

    // Box coloring: cols = x, rows = y, stripes = z

    local_int_t num_color_cols = nx / bx;
    local_int_t num_color_rows = ny / by;
    local_int_t num_color_faces = nz / bz;

    // How is the vector colored
    local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
    local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color



    //TODO: can be improved by using modulo instead of computing the global x, y, z first
    global_int_t gx0 = px * nx;
    global_int_t gy0 = py * ny;
    global_int_t gz0 = pz * nz;
    local_int_t color_offs_x_local = (bx - gx0 % bx + color_offs_x_global) % bx;
    local_int_t color_offs_y_local = (by - gy0 % by + color_offs_y_global) % by;
    local_int_t color_offs_z_local = (bz - gz0 % bz + color_offs_z_global) % bz;

    num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

    // Iterate over all grid nodes of the current color assigned to this thread group.
    for (local_int_t i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        // Find the (ix,iy,iz) position of the node for this color.
        local_int_t ix = i % num_color_cols;
        local_int_t iy = ((i % (num_color_cols * num_color_rows))) / num_color_cols;
        local_int_t iz = i / (num_color_cols * num_color_rows);

        // Map to full local grid coordinates.
        ix = ix * bx + color_offs_x_local;
        iy = iy * by + color_offs_y_local;
        iz = iz * bz + color_offs_z_local;

        // Compute local and global indices.
        local_int_t li = ix + iy * nx + iz * nx * ny;
        global_int_t gi = local_i_to_global_i(li, nx, ny, nz, gnx, gny, gnz, gi0);
        DataType my_sum = 0.0;

        // Loop over matrix stripes assigned to this lane.
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            global_int_t gj = j_min_i_d[stripe] + gi;
            if (gj>= 0 && gj < gnx * gny * gnz) {
                // Convert gj to halo coordinate hj, which is the memory location of gj in the halo struct.
                local_int_t hj =  global_i_to_halo_i(gj, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
                if(hj>=0 && hj<(nx+2)*(ny+2)*(nz+2))
                    my_sum -= striped_A[li * num_stripes + stripe] * x[hj];
            }
        }

        // Warp-level reduction: sum my_sum across all lanes in the cooperation group.
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            local_int_t hi =  global_i_to_halo_i(gi, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
            DataType diag = striped_A[li * num_stripes + diag_offset];
            if(hi>=0 && hi<(nx+2)*(ny+2)*(nz+2)){
                DataType sum = diag * x[hi] + y[hi] + my_sum;
                x[hi] = sum / diag;  
            }         
        }
        __syncthreads();
    }
}

__global__ void striped_box_coloring_half_SymGS_kernel_opt(
    int cooperation_number,
    int color, int bx, int by, int bz,
    local_int_t nx, local_int_t ny, local_int_t nz,
    int num_stripes, int diag_offset,
    DataType * striped_A,
    DataType * x, DataType * y,
    int px, int py, int pz,
    int dimx, int dimy
){
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t coop_group_id = tid / cooperation_number;
    local_int_t lane = tid % cooperation_number;
    local_int_t num_coop_groups = blockDim.x * gridDim.x / cooperation_number;

    // Box coloring: cols = x, rows = y, stripes = z

    local_int_t num_color_cols = nx / bx;
    local_int_t num_color_rows = ny / by;
    local_int_t num_color_faces = nz / bz;

    // How is the vector colored
    local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
    local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color

    local_int_t color_offs_x_local = (color_offs_x_global + (bx - (bx - px % bx))) % bx;
    local_int_t color_offs_y_local =  (color_offs_y_global + (by - (by - py % by))) % by;
    local_int_t color_offs_z_local =  (color_offs_z_global + (bz - (bz - pz % bz))) % bz;

    num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

    // Iterate over all grid nodes of the current color assigned to this thread group.
    for (local_int_t i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        // Find the (ix,iy,iz) position of the node for this color.
        local_int_t ix = i % num_color_cols;
        local_int_t iy = ((i % (num_color_cols * num_color_rows))) / num_color_cols;
        local_int_t iz = i / (num_color_cols * num_color_rows);

        // Map to full local grid coordinates.
        ix = ix * bx + color_offs_x_local;
        iy = iy * by + color_offs_y_local;
        iz = iz * bz + color_offs_z_local;

        // Compute local and halo indices.
        local_int_t li = ix + iy * nx + iz * nx * ny;
        local_int_t hi = local_i_to_halo_i(li, nx, ny, nz, dimx, dimy);

        DataType my_sum = 0.0;

        // Loop over matrix stripes assigned to this lane.
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            local_int_t hj = j_min_i_d[stripe] + hi;
            my_sum -= striped_A[li * num_stripes + stripe] * x[hj];
        }

        // Warp-level reduction: sum my_sum across all lanes in the cooperation group.
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            DataType diag = striped_A[li * num_stripes + diag_offset];
            DataType sum = diag * x[hi] + y[hi] + my_sum;
            x[hi] = sum / diag;  
        }
        __syncthreads();
    }
}

__global__ void column_major_striped_box_coloring_half_SymGS_kernel(
    int cooperation_number,
    int color, int bx, int by, int bz,
    local_int_t nx, local_int_t ny, local_int_t nz,
    int num_stripes, int diag_offset,
    DataType * A_d,
    DataType * x, DataType * y,
    int px, int py, int pz,
    int dimx, int dimy
){
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t coop_group_id = tid / cooperation_number;
    local_int_t lane = tid % cooperation_number;
    local_int_t num_coop_groups = blockDim.x * gridDim.x / cooperation_number;

    // Box coloring: cols = x, rows = y, stripes = z

    local_int_t num_color_cols = nx / bx;
    local_int_t num_color_rows = ny / by;
    local_int_t num_color_faces = nz / bz;

    // How is the vector colored
    local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
    local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color

    local_int_t color_offs_x_local = (color_offs_x_global + (bx - (bx - px % bx))) % bx;
    local_int_t color_offs_y_local =  (color_offs_y_global + (by - (by - py % by))) % by;
    local_int_t color_offs_z_local =  (color_offs_z_global + (bz - (bz - pz % bz))) % bz;

    num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

    // Iterate over all grid nodes of the current color assigned to this thread group.
    for (local_int_t i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        // Find the (ix,iy,iz) position of the node for this color.
        local_int_t ix = i % num_color_cols;
        local_int_t iy = ((i % (num_color_cols * num_color_rows))) / num_color_cols;
        local_int_t iz = i / (num_color_cols * num_color_rows);

        // Map to full local grid coordinates.
        ix = ix * bx + color_offs_x_local;
        iy = iy * by + color_offs_y_local;
        iz = iz * bz + color_offs_z_local;

        // Compute local and halo indices.
        local_int_t li = ix + iy * nx + iz * nx * ny;
        local_int_t hi = local_i_to_halo_i(li, nx, ny, nz, dimx, dimy);

        DataType my_sum = 0.0;

        // Loop over matrix stripes assigned to this lane.
        local_int_t num_rows = nx * ny * nz;
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            local_int_t hj = j_min_i_d[stripe] + hi;
            my_sum -= A_d[num_rows * stripe + li] * x[hj];
        }

        // Warp-level reduction: sum my_sum across all lanes in the cooperation group.
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            DataType diag = A_d[num_rows * diag_offset + li];
            DataType sum = diag * x[hi] + y[hi] + my_sum;
            x[hi] = sum / diag;  
        }
        __syncthreads();
    }
}

__global__ void blocked_striped_box_coloring_half_SymGS_kernel(
    int cooperation_number,
    int color, int bx, int by, int bz,
    local_int_t nx, local_int_t ny, local_int_t nz,
    int num_stripes, int diag_offset,
    DataType * A_d,
    DataType * x, DataType * y,
    int px, int py, int pz,
    int dimx, int dimy
){
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t coop_group_id = tid / cooperation_number;
    local_int_t lane = tid % cooperation_number;
    local_int_t num_coop_groups = blockDim.x * gridDim.x / cooperation_number;

    // Box coloring: cols = x, rows = y, stripes = z

    local_int_t num_color_cols = nx / bx;
    local_int_t num_color_rows = ny / by;
    local_int_t num_color_faces = nz / bz;

    // How is the vector colored
    local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
    local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color

    local_int_t color_offs_x_local = (color_offs_x_global + (bx - (bx - px % bx))) % bx;
    local_int_t color_offs_y_local =  (color_offs_y_global + (by - (by - py % by))) % by;
    local_int_t color_offs_z_local =  (color_offs_z_global + (bz - (bz - pz % bz))) % bz;

    num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;
    // Iterate over all grid nodes of the current color assigned to this thread group.
    for (local_int_t i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        // Find the (ix,iy,iz) position of the node for this color.
        local_int_t ix = i % num_color_cols;
        local_int_t iy = ((i % (num_color_cols * num_color_rows))) / num_color_cols;
        local_int_t iz = i / (num_color_cols * num_color_rows);

        // Map to full local grid coordinates.
        ix = ix * bx + color_offs_x_local;
        iy = iy * by + color_offs_y_local;
        iz = iz * bz + color_offs_z_local;

        // Compute local and halo indices.
        local_int_t li = ix + iy * nx + iz * nx * ny;
        local_int_t hi = local_i_to_halo_i(li, nx, ny, nz, dimx, dimy);

        DataType my_sum = 0.0;

        // Loop over matrix stripes assigned to this lane.
        local_int_t block_index = (li / warpSize) * warpSize * num_stripes;
        local_int_t elem_index_in_block = li % warpSize;
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            local_int_t hj = j_min_i_d[stripe] + hi;
            my_sum -= A_d[block_index + stripe * warpSize + elem_index_in_block] * x[hj];
        }

        // Warp-level reduction: sum my_sum across all lanes in the cooperation group.
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            DataType diag = A_d[block_index + diag_offset * warpSize + elem_index_in_block];
            DataType sum = diag * x[hi] + y[hi] + my_sum;
            x[hi] = sum / diag;  
        }
        __syncthreads();
    }
}

__global__ void color_wise_striped_box_coloring_half_SymGS_kernel(
    int cooperation_number,
    int color, int bx, int by, int bz,
    local_int_t nx, local_int_t ny, local_int_t nz,
    int num_stripes, int diag_offset,
    DataType * A_d,
    DataType * x, DataType * y,
    int px, int py, int pz,
    int dimx, int dimy
){
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Box coloring: cols = x, rows = y, stripes = z

    local_int_t num_color_cols = nx / bx;
    local_int_t num_color_rows = ny / by;
    local_int_t num_color_faces = nz / bz;

    // How is the vector colored
    local_int_t color_offs_x_global = color % bx; //gives x-xcoordinate of first appearance of color
    local_int_t color_offs_y_global = (color - color_offs_x_global) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    local_int_t color_offs_z_global = (color - color_offs_x_global - bx * color_offs_y_global) / (bx * by); //gives z-coordinate of first appearance of color

    local_int_t color_offs_x_local = (color_offs_x_global + (bx - (bx - px % bx))) % bx;
    local_int_t color_offs_y_local =  (color_offs_y_global + (by - (by - py % by))) % by;
    local_int_t color_offs_z_local =  (color_offs_z_global + (bz - (bz - pz % bz))) % bz;

    num_color_cols = (color_offs_x_local < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y_local < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z_local < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;
    int rounded_num_nodes = ((num_nodes_with_color + 31) / 32) * 32;

    //gate
    if(tid >= num_nodes_with_color){
        return;
    }
    // Iterate over all grid nodes of the current color assigned to this thread group.
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

    DataType my_sum = 0.0;

    // Loop over matrix stripes assigned to this lane.
    A_d += rounded_num_nodes * num_stripes * color;
    local_int_t hj;
    for(int stripe = 0; stripe < diag_offset; stripe ++){
        hj = j_min_i_d[stripe] + hi;
        my_sum -= A_d[tid] * x[hj];
        A_d += rounded_num_nodes;
    }
    DataType diag = A_d[tid];
    hj = j_min_i_d[diag_offset] + hi;
    my_sum -= diag * x[hj];
    A_d += rounded_num_nodes;
    
    for(int stripe = diag_offset+1; stripe < num_stripes; stripe++){
        hj = j_min_i_d[stripe] + hi;
        my_sum -= A_d[tid] * x[hj];
        A_d += rounded_num_nodes;
    }
    
    DataType sum = diag * x[hi] + y[hi] + my_sum;
    x[hi] = sum / diag;

}

/**
 * @brief Host function to perform a full parallel Symmetric Gauss-Seidel (SymGS) iteration
 *        using striped box coloring on a structured 3D grid with multiple GPUs.
 *
 * High-level strategy:
 * - Forward SymGS sweep: process colors 0..max_color, one color per kernel launch.
 * - After each color, exchange halo to synchronize boundary values between domains.
 * - Backward SymGS sweep: process colors max_color..0 (reverse order), again with halo exchange after each.
 * - Final halo sync to ensure all values are up-to-date.
 *
 * Each kernel call only processes a single color (not the entire solve).
 *
 * @param A         Matrix in striped storage.
 * @param x_d       Solution vector (device, with halo).
 * @param b_d       Right-hand side vector (device, with halo).
 * @param problem   Problem geometry and parallel decomposition info.
 */
template <typename T>
void striped_multi_GPU_Implementation<T>::striped_box_coloring_multi_GPU_computeSymGS(
    striped_partial_Matrix<T> & A,
    Halo *x_d, Halo *b_d,
    Problem *problem
){
    // Gather geometry and matrix info
    local_int_t nx = problem->nx;
    local_int_t ny = problem->ny;
    local_int_t nz = problem->nz;
    global_int_t gnx = problem->gnx;
    global_int_t gny = problem->gny;
    global_int_t gnz = problem->gnz;
    global_int_t gi0 = problem->gi0;
    int px = problem->px;
    int py = problem->py;
    int pz = problem->pz;

    local_int_t diag_offset = A.get_diag_index();
    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    DataType * striped_A_d = A.get_values_d();

    assert(diag_offset >= 0);

    // Box coloring parameters (must not violate stencil dependencies)
    int bx = this->bx;
    int by = this->by;
    int bz = this->bz;
    assert(bx >= 2);
    assert(by >= 2);
    assert(bz >= 2);

    int cooperation_number = this->SymGS_cooperation_number;

    // Compute number of colors and max number of nodes per color
    int num_colors = bx * by * bz;
    int max_color = num_colors - 1;
    int max_num_rows_per_color = ceiling_division(nx, bx) * ceiling_division(ny, by) * ceiling_division(nz, bz);
    // Use 512 threads per block (or replace with a named constant if desired)
    const int threads_per_block = 512;
    local_int_t num_blocks = (max_num_rows_per_color + threads_per_block - 1) / threads_per_block;
    //local_int_t num_blocks = std::min(ceiling_division(max_num_rows_per_color, threads_per_block/cooperation_number), MAX_NUM_BLOCKS);

    //move j_min_i to constant memory
    std::vector<local_int_t> j_min_i_h(27);
    CHECK_CUDA(cudaMemcpy(j_min_i_h.data(), A.get_j_min_i_halo_d(), 27 * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpyToSymbol(j_min_i_d, j_min_i_h.data(), 27 * sizeof(local_int_t)));


    // Forward SymGS sweep: process colors 0..max_color
    for(int color = 0; color < num_colors; color++){
        blocked_striped_box_coloring_half_SymGS_kernel<<<num_blocks, threads_per_block>>>(
            cooperation_number,
            color, bx, by, bz,
            nx, ny, nz,
            num_stripes, diag_offset,
            striped_A_d,
            x_d->x_d, b_d->x_d,
            px, py, pz,
            x_d->dimx, x_d->dimy
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        this->ExchangeHalo(x_d, problem);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    //Backward SymGS sweep: process colors max_color..0
    for(int color = max_color; color  >= 0; color--){
        blocked_striped_box_coloring_half_SymGS_kernel<<<num_blocks, threads_per_block>>>(
            cooperation_number,
            color, bx, by, bz,
            nx, ny, nz,
            num_stripes, diag_offset,
            striped_A_d,
            x_d->x_d, b_d->x_d,
            px, py, pz,
            x_d->dimx, x_d->dimy
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        this->ExchangeHalo(x_d, problem);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Explicit template instantiation for DataType
template class striped_multi_GPU_Implementation<DataType>;