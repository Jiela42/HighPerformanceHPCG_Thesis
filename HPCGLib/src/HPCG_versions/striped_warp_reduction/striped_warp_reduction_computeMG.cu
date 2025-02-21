#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"

template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeMG(
    striped_Matrix<T> & A,
    T * r_d, T * x_d
){

    CHECK_CUDA(cudamemset(x_d, 0, A.get_num_rows() * sizeof(T)));
    if(A.coarse_matrix != nullptr){ // go to coarser level if it exists
        // int num_presmoother_steps = A.

    }

}

// template instanciation
template class striped_warp_reduction_Implementation<double>;