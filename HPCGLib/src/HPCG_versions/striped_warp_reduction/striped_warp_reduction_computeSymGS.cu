#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>

__inline__ __device__ void loop_body(int lane, int i, int num_cols, int num_stripes, int * j_min_i, double * striped_A, double * x, double * y, double * shared_diag){
    
    double my_sum = 0.0;
    for (int stripe = lane; stripe < num_stripes; stripe += WARP_SIZE){
        int col = j_min_i[stripe] + i;
        double val = striped_A[i * num_stripes + stripe];
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

__global__ void striped_warp_reduction_SymGS_kernel(
    int num_rows, int num_cols,
    int num_stripes,
    int * j_min_i,
    double * striped_A,
    double * x, double * y
){
    // note that here x is the result vector and y is the input vector

    __shared__ double diag_value[1];
    int lane = threadIdx.x % WARP_SIZE;
    
    // forward pass
    for (int i = 0; i < num_rows; i++){
        loop_body(lane, i, num_cols, num_stripes, j_min_i, striped_A, x, y, diag_value);
    }

    __syncthreads();

    // backward pass
    for (int i = num_rows-1; i >= 0; i--){
        loop_body(lane, i, num_cols, num_stripes, j_min_i, striped_A, x, y, diag_value);
    }
}

template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeSymGS(
    striped_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device   
){

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    int * j_min_i = A.get_j_min_i_d();
    double * striped_A_d = A.get_values_d();
    
    // because this is sequential, we only spawn one warp
    int num_threads = WARP_SIZE;
    int num_blocks = 1;

    int max_iterations = this->max_SymGS_iterations;
    // std::cout << "max_iterations = " << max_iterations << std::endl;
    double norm0 = 1.0;
    double normi = norm0;

    if(max_iterations != 1){
        // compute the original L2 norm
        norm0 = this->L2_norm_for_SymGS(A, x_d, y_d);
    }

    for(int i = 0; i < max_iterations && normi/norm0 > this->SymGS_tolerance; i++){


        striped_warp_reduction_SymGS_kernel<<<num_blocks, num_threads>>>(
            num_rows, num_cols,
            num_stripes,
            j_min_i,
            striped_A_d,
            x_d, y_d
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        if(max_iterations != 1){
            normi = this->L2_norm_for_SymGS(A, x_d, y_d);
        }
    }
    
}

// Explicit instantiation of the template
template class striped_warp_reduction_Implementation<double>;