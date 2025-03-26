#include "HPCG_versions/cusparse.hpp"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>


__global__ void cusparse_SymGS_kernel(
    int num_rows,
    int * A_row_ptr, int * A_col_idx, double * A_values,
    double * x, double * y
){
    // note that here x is the result vector and y is the input vector

    __shared__ double diag_value[1];
    int lane = threadIdx.x % WARP_SIZE;


    // forward pass
    for (int i = 0; i < num_rows; i++){
        double my_sum = 0.0;
        for (int j = A_row_ptr[i] + lane; j < A_row_ptr[i+1]; j += WARP_SIZE){
            int col = A_col_idx[j];
            double val = A_values[j];
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
            double diag = diag_value[0];
            double sum = diag * x[i] + y[i] + my_sum;
            x[i] = sum / diag;           
        }
    }

    __syncthreads();

    // backward pass
        for (int i = num_rows-1; i >= 0; i--){
        double my_sum = 0.0;
        for (int j = A_row_ptr[i] + lane; j < A_row_ptr[i+1]; j += WARP_SIZE){
            int col = A_col_idx[j];
            double val = A_values[j];
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
            double diag = diag_value[0];
            double sum = diag * x[i] + y[i] + my_sum;
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

    int num_rows = A.get_num_rows();

    // because this is sequential, we only spawn one warp
    int num_threads = WARP_SIZE;
    int num_blocks = 1;

    int * A_row_ptr_d = A.get_row_ptr_d();
    int * A_col_idx_d = A.get_col_idx_d();
    T * A_values_d = A.get_values_d();

    int max_iterations = this->max_SymGS_iterations;
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
    
    for(int i = 0; i < this->max_SymGS_iterations && normi/norm0 < this->SymGS_tolerance; i++){

        cusparse_SymGS_kernel<<<num_blocks, num_threads>>>(
            num_rows,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            x_d, y_d
        );
    
        CHECK_CUDA(cudaDeviceSynchronize());

        // compute the L2 norm (only if we need it!)
        if(this->max_SymGS_iterations != 1){
            normi = L2_norm_for_SymGS(A, x_d, y_d);
        }
    }
    

   
}

// Explicit instantiation of the template
template class cuSparse_Implementation<double>;