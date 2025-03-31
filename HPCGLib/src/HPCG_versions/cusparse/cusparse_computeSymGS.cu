#include "HPCG_versions/cusparse.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include <iostream>


__global__ void cusparse_SymGS_kernel(
    local_int_t num_rows,
    local_int_t * A_row_ptr, local_int_t * A_col_idx, DataType * A_values,
    DataType * x, DataType * y
){
    // note that here x is the result vector and y is the input vector

    __shared__ double diag_value[1];
    int lane = threadIdx.x % WARP_SIZE;


    // forward pass
    for (local_int_t i = 0; i < num_rows; i++){
        DataType my_sum = 0.0;
        for (local_int_t j = A_row_ptr[i] + lane; j < A_row_ptr[i+1]; j += WARP_SIZE){
            local_int_t col = A_col_idx[j];
            DataType val = A_values[j];
            my_sum -= val * x[col];
            if(i == col){
                diag_value[0] = val;
            }
        }

        // reduce the my_sum using warp reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            DataType diag = diag_value[0];
            DataType sum = diag * x[i] + y[i] + my_sum;
            x[i] = sum / diag;           
        }
    }

    __syncthreads();

    // backward pass
        for (local_int_t i = num_rows-1; i >= 0; i--){
        DataType my_sum = 0.0;
        for (local_int_t j = A_row_ptr[i] + lane; j < A_row_ptr[i+1]; j += WARP_SIZE){
            local_int_t col = A_col_idx[j];
            DataType val = A_values[j];
            my_sum -= val * x[col];
            if(i == col){
                diag_value[0] = val;
            }
        }

        // reduce the my_sum using warp reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            DataType diag = diag_value[0];
            DataType sum = diag * x[i] + y[i] + my_sum;
            x[i] = sum / diag;
            // printf("x[%d] = %f\n", i, x[i]);           
        }
    }
}

template <typename T>
void cuSparse_Implementation<T>::cusparse_computeSymGS(
    sparse_CSR_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device
        
){

    local_int_t num_rows = A.get_num_rows();

    // because this is sequential, we only spawn one warp
    int num_threads = WARP_SIZE;
    int num_blocks = 1;

    local_int_t* A_row_ptr_d = A.get_row_ptr_d();
    local_int_t * A_col_idx_d = A.get_col_idx_d();
    T * A_values_d = A.get_values_d();

    int max_iterations = this->max_SymGS_iterations;
    // std::cout << "max_iterations = " << max_iterations << std::endl;
    double norm0 = 1.0;
    double normi = norm0;

    if(max_iterations != 1){
        // compute the original L2 norm
        norm0 = L2_norm_for_SymGS(A, x_d, y_d);
    }

    // if(A_row_ptr_d == nullptr || A_col_idx_d == nullptr || A_values_d == nullptr){
    //     throw std::runtime_error("The matrix is not allocated on the device");
    // }

    // if(x_d == nullptr || y_d == nullptr){
    //     throw std::runtime_error("The vectors are not allocated on the device");
    // }

    // std::cout << "normi/norm0 = " << normi/norm0 << std::endl;
    // std::cout << "SymGS_tolerance = " << this->SymGS_tolerance << std::endl;

    // bool stop_condition = normi/norm0 < this->SymGS_tolerance;

    // std::cout << "stop condition: " << stop_condition << std::endl;
    
    for(int i = 0; i < max_iterations && normi/norm0 > this->SymGS_tolerance; i++){

        cusparse_SymGS_kernel<<<num_blocks, num_threads>>>(
            num_rows,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            x_d, y_d
        );
        // std::cout << "Iteration " << i << std::endl;
    
        CHECK_CUDA(cudaDeviceSynchronize());

        // compute the L2 norm (only if we need it!)
        if(max_iterations != 1){
            normi = L2_norm_for_SymGS(A, x_d, y_d);
        }
    }
    

   
}

// Explicit instantiation of the template
template class cuSparse_Implementation<double>;