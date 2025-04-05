#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
// #include <cuda_runtime.h>

template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeMG(
    striped_partial_Matrix<T> & A,
    Halo * r_d, Halo * x_d,
    Problem *problem
){ 
    SetHaloZeroGPU(x_d);

    if(A.get_coarse_Matrix() != nullptr){ // go to coarser level if it exists
        local_int_t num_coarse_rows = A.get_coarse_Matrix()->get_num_rows();
        int num_presmoother_steps = A.get_num_MG_pre_smooth_steps();

        for(int i = 0; i < num_presmoother_steps; i++){
            this->compute_SymGS(A, x_d, r_d, problem);
            this->ExchangeHalo(x_d, problem);
        }

        Halo *Axf_d = A.get_coarse_Matrix()->get_Axf_d(); //finer number of rows, needs to be halo bc SPMV
        Halo *rc_d = A.get_coarse_Matrix()->get_rc_d(); //coarse number of rows, needs to be halo bc of recursive call
        Halo *xc_d = A.get_coarse_Matrix()->get_xc_d(); //coarse number of rows, needs to be halo bc of recursive call

        this->compute_SPMV(A, x_d, Axf_d, problem);
        this->ExchangeHalo(Axf_d, problem);

        int num_threads = 1024;
        local_int_t num_blocks = std::max(num_coarse_rows / num_threads, (local_int_t) 1);

        compute_restriction_multi_GPU_kernel<<<num_blocks, num_threads>>>(
            num_coarse_rows,
            Axf_d->x_d, //Halo
            r_d->x_d, //Halo
            rc_d->x_d, //Halo
            A.get_coarse_Matrix()->get_f2c_op_d(), //normal
            r_d->nx, r_d->ny, r_d->nz
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        
        
        Problem coarse_problem;
        GenerateProblem(problem->npx, problem->npy, problem->npz, problem->nx / 2, problem->ny / 2, problem->nz / 2, problem->size, problem->rank, &coarse_problem);
        
        this->ExchangeHalo(rc_d, &coarse_problem);
        this->ExchangeHalo(xc_d, &coarse_problem);
        
        this->compute_MG(*(A.get_coarse_Matrix()), rc_d, xc_d, &coarse_problem);

        this->ExchangeHalo(xc_d, &coarse_problem);
        

        compute_prolongation_multi_GPU_kernel<<<num_blocks, num_threads>>>(
            num_coarse_rows,
            xc_d-> x_d,
            x_d->x_d,
            A.get_coarse_Matrix()->get_f2c_op_d(),
            x_d->nx, x_d->ny, x_d->nz
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        this->ExchangeHalo(x_d, problem);

        int num_post_smoother_steps = A.get_num_MG_post_smooth_steps();
        for(int i = 0; i < num_post_smoother_steps; i++){
            this->compute_SymGS(A, x_d, r_d, problem);
            this->ExchangeHalo(x_d, problem);
        }
    
    } else {
        this->compute_SymGS(A, x_d, r_d, problem);
        this->ExchangeHalo(x_d, problem);
    }
    CHECK_CUDA(cudaDeviceSynchronize());


}

// template instanciation
template class striped_multi_GPU_Implementation<DataType>;