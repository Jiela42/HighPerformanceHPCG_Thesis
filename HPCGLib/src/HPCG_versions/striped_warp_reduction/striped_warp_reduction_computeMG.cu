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

        T* Axf_d = A.get_coarse_Matrix()->get_Axf_d();
        T* rc_d = A.get_coarse_Matrix()->get_rc_d();
        T* xc_d = A.get_coarse_Matrix()->get_xc_d();

        this->compute_SPMV(A, x_d, Axf_d);

        int num_threads = 1024;
        int num_blocks = std::max(num_coarse_rows / num_threads, 1);

        // std::cout << "num_coarse_rows: " << num_coarse_rows << std::endl;
        // std::cout << "num_rows: " << A.get_num_rows() << std::endl;

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
    
    } else {
        this->compute_SymGS(A, x_d, r_d);
    }

}

// template instanciation
template class striped_warp_reduction_Implementation<double>;