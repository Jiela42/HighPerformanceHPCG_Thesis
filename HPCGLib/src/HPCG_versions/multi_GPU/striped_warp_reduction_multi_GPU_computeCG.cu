#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"


template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeCG(
    striped_Matrix<T> & A,
    T * b_d, T * x_d,
    int & n_iters, T& normr, T& normr0
){
    // std::cout << "Running CG with striped warp reduction" << std::endl;
    normr = 0.0;
    double rtz = 0.0;
    double rtz_old = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double pAp = 0.0;

    int nrows = A.get_num_rows();

    // // print tolerance and max iterations for sanity check
    // std::cout << "CG tolerance: " << this->CG_tolerance << std::endl;
    // std::cout << "Max CG iterations: " << this->max_CG_iterations << std::endl;

    // allocate device memory for p, z, Ap,
    // we also need a device copy of normr, pAp, rtz because the dot product is done on the device and writes the result to the device
    T * p_d;
    T * z_d;
    T * Ap_d;
    T * r_d;
    double * normr_d;
    double * pAp_d;
    double * rtz_d;

    CHECK_CUDA(cudaMalloc(&p_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&z_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&Ap_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&r_d, nrows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&normr_d, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&pAp_d, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&rtz_d, sizeof(double)));

    CHECK_CUDA(cudaMemset(normr_d, 0, sizeof(double)));

    // std::vector<T> looki(5);
    

    // std::cout << "normr_d: " << normr_d << std::endl;

    //this->compute_WAXPBY(A, x_d, x_d, p_d, 1.0, 0.0); // p = x
    this->compute_SPMV(A, p_d, Ap_d); //Ap = A*p
    //this->compute_WAXPBY(A, b_d, Ap_d, r_d, 1.0, -1.0); // r = b - Ax (x stored in p)
    this->compute_Dot(A, r_d, r_d, normr_d);
    CHECK_CUDA(cudaMemcpy(&normr, normr_d, sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "Initial residual: " << normr << std::endl;
    normr = sqrt(normr);

    // Record initial residual for convergence testing
    normr0 = normr;

    // Start iterations
    for(int k = 1; k <= this->max_CG_iterations && normr/normr0 > this->CG_tolerance; k++){
        
        if(this->doPreconditioning){
            this->compute_MG(A, r_d, z_d); // Apply preconditioner
        } else {
            //this->compute_WAXPBY(A, r_d, r_d, z_d, 1.0, 0.0); // z = r
        }

        if(k == 1){
            //this->compute_WAXPBY(A, z_d, z_d, p_d, 1.0, 0.0); // Copy Mr to p
            this->compute_Dot(A, r_d, z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(cudaMemcpy(&rtz, rtz_d, sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            rtz_old = rtz;
            this->compute_Dot(A, r_d, z_d, rtz_d); // rtz = r'*z
            CHECK_CUDA(cudaMemcpy(&rtz, rtz_d, sizeof(double), cudaMemcpyDeviceToHost));
            beta = rtz/rtz_old;
            //this->compute_WAXPBY(A, z_d, p_d, p_d, 1.0, beta); // p = z + beta*p
        }

        this->compute_SPMV(A, p_d, Ap_d); // Ap = A*p
        this->compute_Dot(A, p_d, Ap_d, pAp_d); // pAp = p'*Ap
        CHECK_CUDA(cudaMemcpy(&pAp, pAp_d, sizeof(double), cudaMemcpyDeviceToHost));
        alpha = rtz/pAp;
        //this->compute_WAXPBY(A, x_d, p_d, x_d, 1.0, alpha); // x = x + alpha*p
        //this->compute_WAXPBY(A, r_d, Ap_d, r_d, 1.0, -alpha); // r = r - alpha*Ap
        this->compute_Dot(A, r_d, r_d, normr_d); // normr = r'*r
        CHECK_CUDA(cudaMemcpy(&normr, normr_d, sizeof(double), cudaMemcpyDeviceToHost));
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
template class striped_multi_GPU_Implementation<DataType>;

