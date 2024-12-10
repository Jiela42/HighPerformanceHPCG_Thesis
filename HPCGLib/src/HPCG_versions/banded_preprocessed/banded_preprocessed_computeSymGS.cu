#include "HPCG_versions/banded_preprocessed.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include <iostream>
#include <cuda_runtime.h>

#define BAND_PREPED_SYMGS_COOP_NUM 16

__device__ void forward_loop_body(int lane, int i, int num_cols, int num_bands, int diag_index, int * j_min_i, double * banded_A, double * x, double * y, double * shared_diag){
    
    double my_sum = 0.0;
    for (int band = lane; band < diag_index; band += BAND_PREPED_SYMGS_COOP_NUM){
        int col = j_min_i[band] + i;
        double val = banded_A[i * num_bands + band];
        if (col < num_cols && col >= 0){
            // printf("Col: %d\n", col);
            my_sum -= val * x[col];

        }
        if(i == col){
            shared_diag[0] = val;
        }
    }

    // reduce the my_sum using warp reduction
    for (int offset = BAND_PREPED_SYMGS_COOP_NUM/2; offset > 0; offset /= 2){
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    __syncthreads();

    if (lane == 0){
        double diag = shared_diag[0];
        // printf("diag_index: %d, diag: %f\n", diag_index, diag);

        double sum = diag * x[i] + y[i] + my_sum;
        x[i] = sum / diag;           
    }
    __syncthreads();
    
}


__device__ void backward_loop_body(int lane, int i, int num_cols, int num_bands, int diag_index, int * j_min_i, double * banded_A, double * x, double * y, double * shared_diag){
    
    double my_sum = 0.0;
    for (int band = lane + diag_index+1; band < num_bands; band += BAND_PREPED_SYMGS_COOP_NUM){
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
    for (int offset = BAND_PREPED_SYMGS_COOP_NUM/2; offset > 0; offset /= 2){
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

// this does half a matrix-vector multiplication (only the upper diag parts are multiplied)
__global__ void preprocessing_forward(
    int num_rows, int num_cols,
    int num_bands, int diag_index,
    int min_row, int max_row,
    int * j_min_i, double * banded_A,
    double * x, double * y){
    
    // calculate Ax = y

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int lane = threadIdx.x % BAND_PREPED_SYMGS_COOP_NUM;
    
    for (int i = tid/BAND_PREPED_SYMGS_COOP_NUM + min_row; i < max_row; i += (blockDim.x * gridDim.x)/BAND_PREPED_SYMGS_COOP_NUM) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int band = lane + diag_index; band < num_bands; band += BAND_PREPED_SYMGS_COOP_NUM) {
            int j = i + j_min_i[band];
            int current_row = i * num_bands;
            if (j >= 0 && j < num_rows) {
                sum_i += banded_A[current_row + band] * x[j];
            }
        }

        // now let's reduce the sum_i to a single value using warp-level reduction
        for(int offset = BAND_PREPED_SYMGS_COOP_NUM/2; offset > 0; offset /= 2){
            sum_i += __shfl_down_sync(0xFFFFFFFF, sum_i, offset);
        }

        __syncthreads();

        if (lane == 0){
            y[i] = sum_i;
        }
    }
}

__global__ void preprocessing_backward(
    int num_rows, int num_cols,
    int num_bands, int diag_index,
    int min_row, int max_row,
    int * j_min_i, double * banded_A,
    double * x, double * y){
    
     // calculate Ax = y

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int lane = threadIdx.x % BAND_PREPED_SYMGS_COOP_NUM;
    
    for (int i = tid/BAND_PREPED_SYMGS_COOP_NUM + min_row; i < max_row; i += (blockDim.x * gridDim.x)/BAND_PREPED_SYMGS_COOP_NUM) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int band = lane; band <= diag_index; band += BAND_PREPED_SYMGS_COOP_NUM) {
            int j = i + j_min_i[band];
            int current_row = i * num_bands;
            if (j >= 0 && j < num_rows) {
                sum_i += banded_A[current_row + band] * x[j];
            }
        }

        // now let's reduce the sum_i to a single value using warp-level reduction
        for(int offset = BAND_PREPED_SYMGS_COOP_NUM/2; offset > 0; offset /= 2){
            sum_i += __shfl_down_sync(0xFFFFFFFF, sum_i, offset);
        }

        __syncthreads();

        if (lane == 0){
            y[i] = sum_i;
        }
    }
}

__global__ void banded_forward_SymGS(
    int num_rows, int num_cols,
    int num_bands, int diag_index,
    int min_row, int max_row,
    int * j_min_i,
    double * banded_A,
    double * x, double * y
){
    __shared__ double diag_value[1];
    for(int i = min_row; i < max_row; i++){
        forward_loop_body(threadIdx.x, i, num_cols, diag_index, num_bands, j_min_i, banded_A, x, y, diag_value);
    }
}

__global__ void banded_backward_SymGS(
    int num_rows, int num_cols,
    int num_bands, int diag_index,
    int min_row, int max_row,
    int * j_min_i,
    double * banded_A,
    double * x, double * y
){
    __shared__ double diag_value[1];
    for(int i = max_row-1; i >= min_row; i--){
        backward_loop_body(threadIdx.x, i, num_cols, num_bands, diag_index, j_min_i, banded_A, x, y, diag_value);
    }
}

template <typename T>
void banded_preprocessed_Implementation<T>::banded_preprocessed_computeSymGS(
    banded_Matrix<T> & A, // we pass A for the metadata
    T * banded_A_d, // the data of matrix A is already on the device
    int num_rows, int num_cols,
    int num_bands, // the number of bands in the banded matrix
    int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
    T * x_d, T * y_d // the vectors x and y are already on the device
        
){
    int diag_index = A.get_diag_index();

    assert(num_rows == A.get_num_rows());
    assert(num_cols == A.get_num_cols());
    assert(diag_index >= 0);

    int num_rows_while_preprocessing = 2;

    // because this is sequential, we only spawn one warp
    int num_threads = WARP_SIZE;
    int num_blocks = 1;

    int num_threads_preprocessing = 1024;
    int num_blocks_preprocessing = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, BAND_PREPED_SYMGS_COOP_NUM));


    // forward pass
    // first we do preprocessing and a few rows simultaneously

    preprocessing_forward<<<num_blocks_preprocessing, num_threads_preprocessing>>>(
        num_rows, num_cols,
        num_bands, diag_index,
        num_rows_while_preprocessing, num_rows,
        j_min_i, banded_A_d,
        y_d, x_d
    );

    banded_forward_SymGS<<<num_blocks, num_threads>>>(
        num_rows, num_cols,
        num_bands, diag_index,
        0, num_rows_while_preprocessing,
        j_min_i, banded_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    // do the bulk of the forward pass
    banded_forward_SymGS<<<num_blocks, num_threads>>>(
        num_rows, num_cols,
        num_bands, diag_index,
        num_rows_while_preprocessing, num_rows,
        j_min_i, banded_A_d,
        x_d, y_d
    );

    // synchronize the device
    CHECK_CUDA(cudaDeviceSynchronize());


    // backward pass
    // first we do preprocessing and a few rows simultaneously
    int num_rows_left = num_rows - num_rows_while_preprocessing;

    preprocessing_backward<<<num_blocks_preprocessing, num_threads_preprocessing>>>(
        num_rows, num_cols,
        num_bands, diag_index,
        0, num_rows_left,
        j_min_i, banded_A_d,
        y_d, x_d
    );

    banded_backward_SymGS<<<num_blocks, num_threads>>>(
        num_rows_left, num_cols,
        num_bands, diag_index,
        num_rows_left, num_rows,
        j_min_i, banded_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    // do the bulk of the backward pass
    banded_backward_SymGS<<<num_blocks, num_threads>>>(
        num_rows_left, num_cols,
        num_bands, diag_index,
        0, num_rows_left,
        j_min_i, banded_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());
   
}

// Explicit instantiation of the template
template class banded_preprocessed_Implementation<double>;