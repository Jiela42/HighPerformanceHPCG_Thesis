#include "HPCG_versions/striped_preprocessed.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include <iostream>
#include <cuda_runtime.h>

#define BAND_PREPED_SYMGS_COOP_NUM 16

__device__ void forward_loop_body(int lane, local_int_t i, local_int_t num_cols, int num_stripes, int diag_index, local_int_t * j_min_i, double * striped_A, DataType * x, double * y, double * shared_diag){
    
    double my_sum = 0.0;
    for (int stripe = lane; stripe < diag_index; stripe += BAND_PREPED_SYMGS_COOP_NUM){
        local_int_t col = j_min_i[stripe] + i;
        double val = striped_A[i * num_stripes + stripe];
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


__device__ void backward_loop_body(int lane, local_int_t i, local_int_t num_cols, int num_stripes, int diag_index, local_int_t * j_min_i, double * striped_A, double * x, double * y, double * shared_diag){
    
    double my_sum = 0.0;
    for (int stripe = lane + diag_index+1; stripe < num_stripes; stripe += BAND_PREPED_SYMGS_COOP_NUM){
        local_int_t col = j_min_i[stripe] + i;
        double val = striped_A[i * num_stripes + stripe];
        if (col < num_cols && col >= 0){
            my_sum -= val * x[col];
        }
        if(i == col){
            shared_diag[0] = val;
            // printf("row %d\n", i);
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
    local_int_t num_rows, local_int_t num_cols,
    int num_stripes, int diag_index,
    local_int_t min_row, local_int_t max_row,
    local_int_t * j_min_i, DataType * striped_A,
    double * x, double * y){
    
    // calculate Ax = y

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int lane = threadIdx.x % BAND_PREPED_SYMGS_COOP_NUM;
    
    for (local_int_t i = tid/BAND_PREPED_SYMGS_COOP_NUM + min_row; i < max_row; i += (blockDim.x * gridDim.x)/BAND_PREPED_SYMGS_COOP_NUM) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int stripe = lane + diag_index; stripe < num_stripes; stripe += BAND_PREPED_SYMGS_COOP_NUM) {
            local_int_t j = i + j_min_i[stripe];
            local_int_t current_row = i * num_stripes;
            if (j >= 0 && j < num_rows) {
                sum_i += striped_A[current_row + stripe] * x[j];
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
    local_int_t num_rows, local_int_t num_cols,
    int num_stripes, int diag_index,
    local_int_t min_row, local_int_t max_row,
    local_int_t * j_min_i, double * striped_A,
    double * x, double * y){
    
     // calculate Ax = y

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int lane = threadIdx.x % BAND_PREPED_SYMGS_COOP_NUM;
    
    for (local_int_t i = tid/BAND_PREPED_SYMGS_COOP_NUM + min_row; i < max_row; i += (blockDim.x * gridDim.x)/BAND_PREPED_SYMGS_COOP_NUM) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int stripe = lane; stripe <= diag_index; stripe += BAND_PREPED_SYMGS_COOP_NUM) {
            local_int_t j = i + j_min_i[stripe];
            local_int_t current_row = i * num_stripes;
            if (j >= 0 && j < num_rows) {
                sum_i += striped_A[current_row + stripe] * x[j];
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

__global__ void striped_forward_SymGS(
    local_int_t num_rows, local_int_t num_cols,
    int num_stripes, int diag_index,
    local_int_t min_row, local_int_t max_row,
    local_int_t * j_min_i,
    double * striped_A,
    double * x, double * y
){
    __shared__ double diag_value[1];
    for(local_int_t i = min_row; i < max_row; i++){
        forward_loop_body(threadIdx.x, i, num_cols, diag_index, num_stripes, j_min_i, striped_A, x, y, diag_value);
    }
}

__global__ void striped_backward_SymGS(
    local_int_t num_rows, local_int_t num_cols,
    int num_stripes, int diag_index,
    local_int_t min_row, local_int_t max_row,
    local_int_t * j_min_i,
    double * striped_A,
    double * x, double * y
){
    __shared__ double diag_value[1];
    for(local_int_t i = max_row-1; i >= min_row; i--){
        backward_loop_body(threadIdx.x, i, num_cols, num_stripes, diag_index, j_min_i, striped_A, x, y, diag_value);
    }
}

template <typename T>
void striped_preprocessed_Implementation<T>::striped_preprocessed_computeSymGS(
    striped_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device
        
){
    std::vector<T> looki(5);

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    local_int_t * j_min_i = A.get_j_min_i_d();
    T * striped_A_d = A.get_values_d();

    int diag_index = A.get_diag_index();

    int num_rows_while_preprocessing = 4;

    // because this is sequential, we only spawn one warp
    int num_threads = WARP_SIZE;
    int num_blocks = 1;

    int num_threads_preprocessing = 1024;
    int num_blocks_preprocessing = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, BAND_PREPED_SYMGS_COOP_NUM));


    // forward pass
    // first we do preprocessing and a few rows simultaneously

    preprocessing_forward<<<num_blocks_preprocessing, num_threads_preprocessing>>>(
        num_rows, num_cols,
        num_stripes, diag_index,
        num_rows_while_preprocessing, num_rows,
        j_min_i, striped_A_d,
        y_d, x_d
    );

    striped_forward_SymGS<<<num_blocks, num_threads>>>(
        num_rows, num_cols,
        num_stripes, diag_index,
        0, num_rows_while_preprocessing,
        j_min_i, striped_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    // do the bulk of the forward pass
    striped_forward_SymGS<<<num_blocks, num_threads>>>(
        num_rows, num_cols,
        num_stripes, diag_index,
        num_rows_while_preprocessing, num_rows,
        j_min_i, striped_A_d,
        x_d, y_d
    );

    // synchronize the device
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(looki.data(), x_d, 5*sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "after forward pass looki[0]: " << looki[0] << std::endl;


    // backward pass
    // first we do preprocessing and a few rows simultaneously
    local_int_t num_rows_left = num_rows - num_rows_while_preprocessing;

    preprocessing_backward<<<num_blocks_preprocessing, num_threads_preprocessing>>>(
        num_rows, num_cols,
        num_stripes, diag_index,
        0, num_rows_left,
        j_min_i, striped_A_d,
        y_d, x_d
    );

    striped_backward_SymGS<<<num_blocks, num_threads>>>(
        num_rows_left, num_cols,
        num_stripes, diag_index,
        num_rows_left, num_rows,
        j_min_i, striped_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(looki.data(), x_d, 5*sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "after backward preprocessing pass looki[0]: " << looki[0] << std::endl;

    std::cout << "num_rows_left: " << num_rows_left << std::endl;

    // do the bulk of the backward pass
    striped_backward_SymGS<<<num_blocks, num_threads>>>(
        num_rows_left, num_cols,
        num_stripes, diag_index,
        0, num_rows_left,
        j_min_i, striped_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(looki.data(), x_d, 5*sizeof(T), cudaMemcpyDeviceToHost));

    std::cout << "at end looki[0]: " << looki[0] << std::endl;
   
}

// Explicit instantiation of the template
template class striped_preprocessed_Implementation<double>;