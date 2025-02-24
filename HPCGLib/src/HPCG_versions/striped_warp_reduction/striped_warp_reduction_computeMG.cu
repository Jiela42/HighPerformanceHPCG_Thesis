#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
// #include <cuda_runtime.h>

template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeMG(
    striped_Matrix<T> & A,
    T * r_d, T * x_d
){

    CHECK_CUDA(cudaMemset(x_d, 0, A.get_num_rows() * sizeof(T)));

    if(A.get_coarse_Matrix() != nullptr){ // go to coarser level if it exists
        int num_coarse_rows = A.get_coarse_Matrix()->get_num_rows();
        int num_presmoother_steps = A.get_num_MG_pre_smooth_steps();

        for(int i = 0; i < num_presmoother_steps; i++){
            this->compute_SymGS(A, x_d, r_d);
        }

        T* Axf_d;
        T* rc_d;
        T* xc_d;
        CHECK_CUDA(cudaMalloc(&Axf_d, A.get_num_rows() * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&rc_d, num_coarse_rows * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&xc_d, num_coarse_rows * sizeof(T)));

        CHECK_CUDA(cudaMemset(rc_d, 0, num_coarse_rows * sizeof(T)));
        CHECK_CUDA(cudaMemset(xc_d, 0, num_coarse_rows * sizeof(T)));

        this->compute_SPMV(A, x_d, Axf_d);

        int num_threads = 1024;
        int num_blocks = std::max(num_coarse_rows / num_threads, 1);

        compute_restriction_kernel<<<num_blocks, num_threads>>>(
            num_coarse_rows,
            Axf_d,
            r_d,
            rc_d,
            A.get_coarse_Matrix()->get_f2c_op_d()
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        this->compute_MG(*(A.get_coarse_Matrix()), rc_d, xc_d);

        compute_prolongation_kernel<<<num_blocks, num_threads>>>(
            num_coarse_rows,
            xc_d,
            x_d,
            A.get_coarse_Matrix()->get_f2c_op_d()
        );
        CHECK_CUDA(cudaDeviceSynchronize());


        int num_post_smoother_steps = A.get_num_MG_post_smooth_steps();
        for(int i = 0; i < num_post_smoother_steps; i++){
            this->compute_SymGS(A, x_d, r_d);
        }

        // if we really don't put the stuff into the matrix, we gotta free it here
        CHECK_CUDA(cudaFree(Axf_d));
        CHECK_CUDA(cudaFree(rc_d));
        CHECK_CUDA(cudaFree(xc_d));    
    } else {
        this->compute_SymGS(A, x_d, r_d);
    }

}

// template instanciation
template class striped_warp_reduction_Implementation<double>;