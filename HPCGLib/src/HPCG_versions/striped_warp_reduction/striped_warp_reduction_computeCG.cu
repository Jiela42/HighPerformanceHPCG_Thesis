#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/cuda_utils.hpp"


template <typename T>
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeCG(
    striped_Matrix<T> & A,
    T * b_d, T * x_d,
    int & n_iters, T& normr, T& normr0
){
    // std::cout << "Running CG with striped warp reduction" << std::endl;
    normr = 0.0;
    T rtz = 0.0;
    T rtz_old = 0.0;
    T alpha = 0.0;
    T beta = 0.0;
    T pAp = 0.0;

    local_int_t nrows = A.get_num_rows();

    // // print tolerance and max iterations for sanity check
    // std::cout << "CG tolerance: " << this->CG_tolerance << std::endl;
    // std::cout << "Max CG iterations: " << this->max_CG_iterations << std::endl;

    // allocate device memory for p, z, Ap,
    // we also need a device copy of normr, pAp, rtz because the dot product is done on the device and writes the result to the device
    T * p_d;
    T * z_d;
    T * Ap_d;
    T * r_d;
    T * normr_d;
    T * pAp_d;
    T * rtz_d;

    CHECK_CUDA(cudaMalloc(&p_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&z_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&Ap_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&r_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&normr_d, sizeof(T)));
    CHECK_CUDA(cudaMalloc(&pAp_d, sizeof(T)));
    CHECK_CUDA(cudaMalloc(&rtz_d, sizeof(T)));

    CHECK_CUDA(cudaMemset(normr_d, 0, sizeof(T)));

    // std::vector<T> looki(5);
    // CHECK_CUDA(cudaMemcpy(looki.data(), b_d, 5*sizeof(T), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < 5; i++){
    //     std::cout << "y[" << i << "]: " << looki[i] << std::endl;
    // }
    

    // std::cout << "normr_d: " << normr_d << std::endl;

    this->compute_WAXPBY(A, x_d, x_d, p_d, 1.0, 0.0); // p = x
    this->compute_SPMV(A, p_d, Ap_d); //Ap = A*p
    this->compute_WAXPBY(A, b_d, Ap_d, r_d, 1.0, -1.0); // r = b - Ax (x stored in p)
    this->compute_Dot(A, r_d, r_d, normr_d);
    CHECK_CUDA(cudaMemcpy(&normr, normr_d, sizeof(T), cudaMemcpyDeviceToHost));
    // std::cout << "Initial residual: " << normr << std::endl;
    normr = sqrt(normr);

    // Record initial residual for convergence testing
    normr0 = normr;

    // std::cout << "norm quotient: " << normr/normr0 << std::endl;

    // Start iterations
    for(int k = 1; k <= this->max_CG_iterations && normr/normr0 > this->CG_tolerance; k++){
        
        if(this->doPreconditioning){
            this->compute_MG(A, r_d, z_d); // Apply preconditioner
        } else {
            this->compute_WAXPBY(A, r_d, r_d, z_d, 1.0, 0.0); // z = r
        }

        if(k == 1){
            this->compute_WAXPBY(A, z_d, z_d, p_d, 1.0, 0.0); // Copy Mr to p
            this->compute_Dot(A, r_d, z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(cudaMemcpy(&rtz, rtz_d, sizeof(T), cudaMemcpyDeviceToHost));
        } else {
            rtz_old = rtz;
            this->compute_Dot(A, r_d, z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(cudaMemcpy(&rtz, rtz_d, sizeof(T), cudaMemcpyDeviceToHost));
            beta = rtz/rtz_old;
            this->compute_WAXPBY(A, z_d, p_d, p_d, 1.0, beta); // p = z + beta*p
        }

        this->compute_SPMV(A, p_d, Ap_d); // Ap = A*p
        this->compute_Dot(A, p_d, Ap_d, pAp_d); // pAp = p'*Ap
        CHECK_CUDA(cudaMemcpy(&pAp, pAp_d, sizeof(T), cudaMemcpyDeviceToHost));
        alpha = rtz/pAp;
        this->compute_WAXPBY(A, x_d, p_d, x_d, 1.0, alpha); // x = x + alpha*p
        this->compute_WAXPBY(A, r_d, Ap_d, r_d, 1.0, -alpha); // r = r - alpha*Ap
        this->compute_Dot(A, r_d, r_d, normr_d); // normr = r'*r
        CHECK_CUDA(cudaMemcpy(&normr, normr_d, sizeof(T), cudaMemcpyDeviceToHost));
        normr = sqrt(normr);
        n_iters = k;
    }

    // Free device memory
    CHECK_CUDA(cudaFree(p_d));
    CHECK_CUDA(cudaFree(z_d));
    CHECK_CUDA(cudaFree(Ap_d));
    CHECK_CUDA(cudaFree(r_d));
    CHECK_CUDA(cudaFree(normr_d));
    CHECK_CUDA(cudaFree(pAp_d));
    CHECK_CUDA(cudaFree(rtz_d));

    // std::cout << "CG converged in " << n_iters << " iterations" << std::endl;

}

// template instanciation
template class striped_warp_reduction_Implementation<DataType>;

