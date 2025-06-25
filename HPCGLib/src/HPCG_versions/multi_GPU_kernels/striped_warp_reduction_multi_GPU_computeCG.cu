// This file implements the Conjugate Gradient (CG) solver on multiple GPUs using striped matrix storage,
// warp-level reductions, and optional multigrid preconditioning with halo-structured vector updates.

#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"

/**
 * @brief Performs the Conjugate Gradient method to solve Ax = b using a multi-GPU environment.
 *
 * Uses striped matrix format and halo-exchanged vectors. Optionally applies a multigrid preconditioner.
 * CG loop continues until convergence or max iterations.
 *
 * @param A         System matrix in striped partial format.
 * @param b_d       RHS vector (device).
 * @param x_d       Solution vector (device), updated in-place.
 * @param n_iters   Output: number of iterations performed.
 * @param normr     Output: final residual norm.
 * @param normr0    Output: initial residual norm.
 * @param problem   Geometry and halo metadata.
 */
template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeCG(
    striped_partial_Matrix<T> & A,
    Halo * b_d, Halo * x_d,
    int & n_iters, T& normr, T& normr0,
    Problem *problem
){
    normr = 0.0;
    DataType rtz = 0.0;
    DataType rtz_old = 0.0;
    DataType alpha = 0.0;
    DataType beta = 0.0;
    DataType pAp = 0.0;
    
    local_int_t rows = A.get_num_rows();
    
    // --- Initialization of vectors and buffers ---
    // allocate device memory for p, z, Ap, r vectors,
    // and device copies of scalars normr, pAp, rtz for dot product results
    Halo p_d;
    Halo z_d;
    Halo Ap_d;
    Halo r_d;
    DataType * normr_d;
    DataType * pAp_d;
    DataType * rtz_d;
    
    InitHalo(&p_d, problem);
    InitHalo(&z_d, problem);
    InitHalo(&Ap_d, problem);
    InitHalo(&r_d, problem);
    
    CHECK_CUDA(cudaMalloc(&normr_d, sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&pAp_d, sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&rtz_d, sizeof(DataType)));
    CHECK_CUDA(cudaMemset(normr_d, 0, sizeof(DataType)));
    
    // p = x
    this->compute_WAXPBY(x_d, x_d, &p_d, 1.0, 0.0, problem, false);
    this->ExchangeHalo(&p_d, problem);

    
    // Ap = A*p
    this->compute_SPMV(A, &p_d, &Ap_d, problem);
    this->ExchangeHalo(&Ap_d, problem);
    
    // r = b - Ap (with x stored in p)
    this->compute_WAXPBY(b_d, &Ap_d, &r_d, 1.0, -1.0, problem, false);
    this->ExchangeHalo(&r_d, problem);

    // normr = sqrt(r'*r)
    this->compute_Dot(&r_d, &r_d, normr_d);
    CHECK_CUDA(cudaMemcpy(&normr, normr_d, sizeof(DataType), cudaMemcpyDeviceToHost));
    normr = sqrt(normr);
    
    // Record initial residual for convergence testing
    normr0 = normr;
    
    // --- Start iterations ---
    for(int k = 1; normr/normr0 > this->CG_tolerance && k <= this->max_CG_iterations; k++){
        
        // --- Residual and preconditioner steps ---
        if(this->doPreconditioning){
            // Apply preconditioner: z = M*r
            this->compute_MG(A, &r_d, &z_d, problem);
            this->ExchangeHalo(&z_d, problem);
        } else {
            // No preconditioning: z = r
            this->compute_WAXPBY(&r_d, &r_d, &z_d, 1.0, 0.0, problem, false);
            this->ExchangeHalo(&z_d, problem);
        }
        
        if(k == 1){
            // p = z (copy Mr to p)
            this->compute_WAXPBY(&z_d, &z_d, &p_d, 1.0, 0.0, problem, false);
            this->ExchangeHalo(&p_d, problem);
            // rtz = r'*z
            this->compute_Dot(&r_d, &z_d, rtz_d);
            CHECK_CUDA(cudaMemcpy(&rtz, rtz_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        } else {
            // Update beta and p
            rtz_old = rtz;
            this->compute_Dot(&r_d, &z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(cudaMemcpy(&rtz, rtz_d, sizeof(DataType), cudaMemcpyDeviceToHost));
            beta = rtz/rtz_old;
            // p = z + beta*p
            this->compute_WAXPBY(&z_d, &p_d, &p_d, 1.0, beta, problem, false);
            this->ExchangeHalo(&p_d, problem);
        }
        
        // --- Dot product steps and coefficient calculations ---
        // Ap = A*p
        this->compute_SPMV(A, &p_d, &Ap_d, problem);
        this->ExchangeHalo(&Ap_d, problem);
        // pAp = p'*Ap
        this->compute_Dot(&p_d, &Ap_d, pAp_d);
        CHECK_CUDA(cudaMemcpy(&pAp, pAp_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        alpha = rtz/pAp;
        
        // --- Updates to x, r, p ---
        // x = x + alpha*p
        this->compute_WAXPBY(x_d, &p_d, x_d, 1.0, alpha, problem, false);
        this->ExchangeHalo(x_d, problem);
        // r = r - alpha*Ap
        this->compute_WAXPBY(&r_d, &Ap_d, &r_d, 1.0, -alpha, problem, false);
        this->ExchangeHalo(&r_d, problem);
        // normr = sqrt(r'*r)
        this->compute_Dot(&r_d, &r_d, normr_d);
        CHECK_CUDA(cudaMemcpy(&normr, normr_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        normr = sqrt(normr);
        n_iters = k;
        
    }
    
    // --- Memory cleanup ---
    FreeHaloGPU(&p_d);
    FreeHaloCPU(&p_d);
    FreeHaloGPU(&z_d);
    FreeHaloCPU(&z_d);
    FreeHaloGPU(&Ap_d);
    FreeHaloCPU(&Ap_d);
    FreeHaloGPU(&r_d);
    FreeHaloCPU(&r_d);
    CHECK_CUDA(cudaFree(normr_d));
    CHECK_CUDA(cudaFree(pAp_d));
    CHECK_CUDA(cudaFree(rtz_d));

    CHECK_CUDA(cudaDeviceSynchronize());
}

// template instanciation
template class striped_multi_GPU_Implementation<DataType>;
