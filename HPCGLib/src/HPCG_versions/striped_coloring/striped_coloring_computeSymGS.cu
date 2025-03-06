#include "HPCG_versions/striped_coloring.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include "MatrixLib/coloring.cuh"
// #include <iostream>
// #include <cuda_runtime.h>

__global__ void striped_coloring_half_SymGS_kernel(
    int color, int * color_pointer, int * color_sorted_rows,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    double * x, double * y
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    int start = color_pointer[color];
    int end = color_pointer[color+1];

    for(int i = warp_id + start; i < end; i += num_warps){
        int row = color_sorted_rows[i];
            double my_sum = 0.0;
            for(int stripe = lane_id; stripe < num_stripes; stripe += WARP_SIZE){
                int col = j_min_i[stripe] + row;
                double val = striped_A[row * num_stripes + stripe];
                if(col < num_cols && col >= 0){
                    my_sum -= val * x[col];
                }
            }

            // reduce the my_sum using warp reduction
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
                my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
            }

            __syncthreads();
            if (lane_id == 0){
                double diag = striped_A[row * num_stripes + diag_offset];
                double sum = diag * x[row] + y[row] + my_sum;
                x[row] = sum / diag;           
            }
            __syncthreads();
        
    }
}


__global__ void striped_coloring_SymGS_backward_kernel(
    int color, int * colors,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    double * x, double * y
){

     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for(int i = num_rows-1-warp_id; i >= 0; i += num_warps){
        if(colors[i] == color){
            double my_sum = 0.0;
            for(int stripe = lane_id; stripe < num_stripes; stripe += WARP_SIZE){
                int col = j_min_i[stripe] + i;
                double val = striped_A[i * num_stripes + stripe];
                if(col < num_cols && col >= 0){
                    my_sum -= val * x[col];
                }
            }

            // reduce the my_sum using warp reduction
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
                my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
            }

            __syncthreads();
            if (lane_id == 0){
                double diag = striped_A[i * num_stripes + diag_offset];
                double sum = diag * x[i] + y[i] + my_sum;
                x[i] = sum / diag;           
            }
            __syncthreads();
        }
    }
}

__global__ void compute_num_colors_per_row(int num_rows, int max_num_colors, int * colors, int * num_colors_per_row){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = tid; i < max_num_colors; i += gridDim.x * blockDim.x){
        int num_rows_per_i = 0;
        for(int j = 0; j < num_rows; j++){
            if(colors[j] == i){
                num_rows_per_i++;
            }
        }
        num_colors_per_row[i] = num_rows_per_i;
    }
}

template <typename T>
void striped_coloring_Implementation<T>::striped_coloring_computeSymGS(
    striped_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device
){
    int diag_offset = A.get_diag_index();

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    int * j_min_i = A.get_j_min_i_d();
    T * striped_A_d = A.get_values_d();

    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();

    // first we need to color the matrix
    // we make a device vector for the colors
    int * color_pointer_d;
    int * color_sorted_rows_d;

    // we allocate the memory for the colors
    CHECK_CUDA(cudaMalloc(&color_pointer_d, (num_rows+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&color_sorted_rows_d, num_rows * sizeof(int)));
    
    CHECK_CUDA(cudaMemset(color_pointer_d, 0, (num_rows+1) * sizeof(int)));

    // we need to get the colors
    get_color_row_mapping(nx, ny, nz, color_pointer_d, color_sorted_rows_d);

    // the number of blocks is now dependent on the maximum number of rows per color

    int max_num_rows_per_color = std::min(nx * ny / 4, std::min(nx * nz / 2, ny * nz));
    int max_color = (nx-1) + 2 * (ny-1) + 4 * (nz-1);

    int num_blocks = std::min(ceiling_division(max_num_rows_per_color, 1024/WARP_SIZE), MAX_NUM_BLOCKS);
    for(int color = 0; color <= max_color; color++){
        // we need to do a forward pass
        striped_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
        color, color_pointer_d, color_sorted_rows_d,
        num_rows, num_cols,
        num_stripes, diag_offset,
        j_min_i,
        striped_A_d,
        x_d, y_d
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // we need to do a backward pass,
    // the colors for this are the same just in reverse order
    
    for(int color = max_color; color  >= 0; color--){

        striped_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
        color, color_pointer_d, color_sorted_rows_d,
        num_rows, num_cols,
        num_stripes, diag_offset,
        j_min_i,
        striped_A_d,
        x_d, y_d
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // free the memory
    CHECK_CUDA(cudaFree(color_pointer_d));
    CHECK_CUDA(cudaFree(color_sorted_rows_d));
    
}

// explicit template instantiation
template class striped_coloring_Implementation<double>;