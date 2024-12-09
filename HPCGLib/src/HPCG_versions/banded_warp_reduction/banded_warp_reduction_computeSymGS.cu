#include "HPCG_versions/banded_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>

__inline__ __device__ void loop_body(int lane, int i, int num_cols, int num_bands, int * j_min_i, double * banded_A, double * x, double * y, double * shared_diag){
    
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

__global__ void banded_warp_reduction_SymGS_kernel(
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i,
    double * banded_A,
    double * x, double * y
){
    // note that here x is the result vector and y is the input vector

    __shared__ double diag_value[1];
    int lane = threadIdx.x % WARP_SIZE;
    
    // forward pass
    for (int i = 0; i < num_rows; i++){
        loop_body(lane, i, num_cols, num_bands, j_min_i, banded_A, x, y, diag_value);
    }

    __syncthreads();

    // backward pass
    for (int i = num_rows-1; i >= 0; i--){
        loop_body(lane, i, num_cols, num_bands, j_min_i, banded_A, x, y, diag_value);
    }
}

template <typename T>
void banded_warp_reduction_Implementation<T>::banded_warp_reduction_computeSymGS(
    banded_Matrix<T> & A, // we pass A for the metadata
    T * banded_A_d, // the data of matrix A is already on the device
    int num_rows, int num_cols,
    int num_bands, // the number of bands in the banded matrix
    int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
    T * x_d, T * y_d // the vectors x and y are already on the device
        
){

    assert(num_rows == A.get_num_rows());
    assert(num_cols == A.get_num_cols());
    
    // because this is sequential, we only spawn one warp
    int num_threads = WARP_SIZE;
    int num_blocks = 1;

    banded_warp_reduction_SymGS_kernel<<<num_blocks, num_threads>>>(
        num_rows, num_cols,
        num_bands,
        j_min_i,
        banded_A_d,
        x_d, y_d
    );

    CHECK_CUDA(cudaDeviceSynchronize());
   
}

// Explicit instantiation of the template
template class banded_warp_reduction_Implementation<double>;