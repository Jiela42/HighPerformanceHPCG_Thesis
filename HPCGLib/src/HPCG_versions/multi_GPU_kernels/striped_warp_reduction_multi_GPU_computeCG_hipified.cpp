#include "HPCG_versions/striped_multi_GPU_hipified.cuh"
#include "UtilLib/cuda_utils_hipified.hpp"


template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeCG(
    striped_partial_Matrix<T> & A,
    Halo * b_d, Halo * x_d,
    int & n_iters, T& normr, T& normr0,
    Problem *problem
){

    
    // std::cout << "Running CG with striped warp reduction" << std::endl;
    normr = 0.0;
    DataType rtz = 0.0;
    DataType rtz_old = 0.0;
    DataType alpha = 0.0;
    DataType beta = 0.0;
    DataType pAp = 0.0;
    
    local_int_t rows = A.get_num_rows();
    
    // // print tolerance and max iterations for sanity check
    // std::cout << "CG tolerance: " << this->CG_tolerance << std::endl;
    // std::cout << "Max CG iterations: " << this->max_CG_iterations << std::endl;
    
    // allocate device memory for p, z, Ap,
    // we also need a device copy of normr, pAp, rtz because the dot product is done on the device and writes the result to the device
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
    
    CHECK_CUDA(hipMalloc(&normr_d, sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&pAp_d, sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&rtz_d, sizeof(DataType)));
    CHECK_CUDA(hipMemset(normr_d, 0, sizeof(DataType)));
    // std::vector<T> looki(5);
    
    
    // std::cout << "normr_d: " << normr_d << std::endl;
    
    this->compute_WAXPBY(x_d, x_d, &p_d, 1.0, 0.0, problem, false); // p = x
    this->ExchangeHalo(&p_d, problem);
    this->compute_SPMV(A, &p_d, &Ap_d, problem); //Ap = A*p
    this->ExchangeHalo(&Ap_d, problem);
    this->compute_WAXPBY(b_d, &Ap_d, &r_d, 1.0, -1.0, problem, false); // r = b - Ax (x stored in p)
    this->ExchangeHalo(&r_d, problem);
    this->compute_Dot(&r_d, &r_d, normr_d);
    CHECK_CUDA(hipMemcpy(&normr, normr_d, sizeof(DataType), hipMemcpyDeviceToHost));
    // std::cout << "Initial residual: " << normr << std::endl;
    normr = sqrt(normr);
    
    // Record initial residual for convergence testing
    normr0 = normr;
    
    // Start iterations
    for(int k = 1; k <= this->max_CG_iterations && normr/normr0 > this->CG_tolerance; k++){
        if(problem->rank == 0) printf("MULTI CG iteration %d \t tolreance=%f\n", k, normr/normr0);
        
        if(this->doPreconditioning){
            this->compute_MG(A, &r_d, &z_d, problem); // Apply preconditioner
        } else {
            this->compute_WAXPBY(&r_d, &r_d, &z_d, 1.0, 0.0, problem, false); // z = r
            this->ExchangeHalo(&z_d, problem);
        }
        
        if(k == 1){
            this->compute_WAXPBY(&z_d, &z_d, &p_d, 1.0, 0.0, problem, false); // Copy Mr to p
            this->ExchangeHalo(&p_d, problem);
            this->compute_Dot(&r_d, &z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(hipMemcpy(&rtz, rtz_d, sizeof(DataType), hipMemcpyDeviceToHost));
        } else {
            rtz_old = rtz;
            this->compute_Dot(&r_d, &z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(hipMemcpy(&rtz, rtz_d, sizeof(DataType), hipMemcpyDeviceToHost));
            beta = rtz/rtz_old;
            this->compute_WAXPBY(&z_d, &p_d, &p_d, 1.0, beta, problem, false); // p = z + beta*p
            this->ExchangeHalo(&p_d, problem);
        }
        
        this->compute_SPMV(A, &p_d, &Ap_d, problem); // Ap = A*p
        this->ExchangeHalo(&Ap_d, problem);
        this->compute_Dot(&p_d, &Ap_d, pAp_d); // pAp = p'*Ap
        CHECK_CUDA(hipMemcpy(&pAp, pAp_d, sizeof(DataType), hipMemcpyDeviceToHost));
        alpha = rtz/pAp;
        this->compute_WAXPBY(x_d, &p_d, x_d, 1.0, alpha, problem, false); // x = x + alpha*p
        this->ExchangeHalo(x_d, problem);
        this->compute_WAXPBY(&r_d, &Ap_d, &r_d, 1.0, -alpha, problem, false); // r = r - alpha*Ap
        this->ExchangeHalo(&r_d, problem);
        this->compute_Dot(&r_d, &r_d, normr_d); // normr = r'*r
        CHECK_CUDA(hipMemcpy(&normr, normr_d, sizeof(DataType), hipMemcpyDeviceToHost));
        normr = sqrt(normr);
        n_iters = k;
    }
    
    // Free device memory
    FreeHaloGPU(&p_d);
    FreeHaloCPU(&p_d);
    FreeHaloGPU(&z_d);
    FreeHaloCPU(&z_d);
    FreeHaloGPU(&Ap_d);
    FreeHaloCPU(&Ap_d);
    FreeHaloGPU(&r_d);
    FreeHaloCPU(&r_d);
    CHECK_CUDA(hipFree(normr_d));
    CHECK_CUDA(hipFree(pAp_d));
    CHECK_CUDA(hipFree(rtz_d));

    // std::cout << "CG converged in " << n_iters << " iterations" << std::endl;

}

// template instanciation
template class striped_multi_GPU_Implementation<DataType>;

