#include "HPCG_versions/banded_coloring.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include "MatrixLib/coloring.cuh"
// #include <iostream>
// #include <cuda_runtime.h>

__global__ void banded_coloring_SymGS_forward_kernel(
    int color, int * colors,
    int num_rows, int num_cols,
    int num_bands, int diag_offset,
    int * j_min_i,
    double * banded_A,
    double * x, double * y
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for(int i = warp_id; i < num_rows; i += num_warps){
        if(colors[i] == color){
            double my_sum = 0.0;
            for(int band = lane_id; band < num_bands; band += WARP_SIZE){
                int col = j_min_i[band] + i;
                double val = banded_A[i * num_bands + band];
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
                double diag = banded_A[i * num_bands + diag_offset];
                double sum = diag * x[i] + y[i] + my_sum;
                x[i] = sum / diag;           
            }
            __syncthreads();
        }
    }
}


__global__ void banded_coloring_SymGS_backward_kernel(
    // int color, int * colors,
    int num_rows, int num_cols,
    int num_bands, int diag_offset,
    int * j_min_i,
    double * banded_A,
    double * x, double * y
){

    int lane = threadIdx.x % WARP_SIZE;
    __shared__ double shared_diag[1];

    for (int i = num_rows-1; i >= 0; i--){
        
        double my_sum = 0.0;
        for (int band = lane; band < num_bands; band += WARP_SIZE){
            int col = j_min_i[band] + i;
            double val = banded_A[i * num_bands + band];
            if (col < num_cols && col >= 0){
                my_sum -= val * x[col];
            }
            if(i == col){
                shared_diag[0] = val;
            }
        }

        // reduce the my_sum using warp reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            double diag = shared_diag[0];
            double sum = diag * x[i] + y[i] + my_sum;
            x[i] = sum / diag;           
        }
        __syncthreads();
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
void banded_coloring_Implementation<T>::banded_coloring_computeSymGS(
    banded_Matrix<T> & A, // we pass A for the metadata
    T * banded_A_d, // the data of matrix A is already on the device
    int num_rows, int num_cols,
    int num_bands, // the number of bands in the banded matrix
    int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
    T * x_d, T * y_d // the vectors x and y are already on the device
){
    int diag_offset = A.get_diag_index();
    assert(num_bands == A.get_num_bands());
    assert(num_rows == A.get_num_rows());
    assert(num_cols == A.get_num_cols());
    assert(diag_offset >= 0);

    // first we need to color the matrix
    // we make a device vector for the colors
    int * colors_d;

    // we allocate the memory for the colors
    CHECK_CUDA(cudaMalloc(&colors_d, num_rows * sizeof(int)));

    // we initialize the colors to -1
    CHECK_CUDA(cudaMemset(colors_d, -1, num_rows * sizeof(int)));

    color_for_forward_pass_kernel<<<1, 1024>>>(num_rows, num_bands, diag_offset, banded_A_d, j_min_i, colors_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    int max_color;
    CHECK_CUDA(cudaMemcpy(&max_color, &colors_d[num_rows-1], sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> num_colors_per_row(max_color, -1);
    // int * num_colors_per_row_d;
    // CHECK_CUDA(cudaMalloc(&num_colors_per_row_d, max_color * sizeof(int)));

    // int num_blocks = std::min(ceiling_division(max_color, 1024));
    // compute_num_colors_per_row<<<num_blocks, 1024>>>(num_rows, max_color, colors_d, num_colors_per_row_d);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaMemcpy(num_colors_per_row.data(), num_colors_per_row_d, max_color * sizeof(int), cudaMemcpyDeviceToHost));

    for(int color = 0; color < max_color; color++){
        // we need to do a forward pass
        int num_blocks = std::min(ceiling_division(num_rows, 1024/WARP_SIZE), MAX_NUM_BLOCKS);
        banded_coloring_SymGS_forward_kernel<<<num_blocks, 1024>>>(
        color, colors_d,
        num_rows, num_cols,
        num_bands, diag_offset,
        j_min_i,
        banded_A_d,
        x_d, y_d
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // we need to do a backward pass
    int num_blocks = 1;
    int num_threads = WARP_SIZE;
    banded_coloring_SymGS_backward_kernel<<<num_blocks, num_threads>>>(
        num_rows, num_cols,
        num_bands, diag_offset,
        j_min_i,
        banded_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());

}

// explicit template instantiation
template class banded_coloring_Implementation<double>;