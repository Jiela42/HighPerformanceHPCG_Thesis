#ifndef STRIPED_WARP_REDUCTION_CUH
#define STRIPED_WARP_REDUCTION_CUH

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_mpi_utils.cuh"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
class striped_warp_reduction_Implementation : public HPCG_functions<T> {
public:

    int dot_cooperation_number; // the cooperation number for the dot product

    striped_warp_reduction_Implementation(){
        // overwritting the inherited variables

        this->dot_cooperation_number = 8;

        this->doPreconditioning = true;

        this->version_name = "Striped Warp Reduction";
        this->implementation_type = Implementation_Type::STRIPED;
        this->SPMV_implemented = true;
        this->Dot_implemented = true;
        this->SymGS_implemented = true;
        this->WAXPBY_implemented = true;
        this->CG_implemented = true;
        this->MG_implemented = true;
        
    }

    void compute_CG(
        striped_Matrix<T> & A,
        T * b_d, T * x_d,
        int & n_iters, T& normr, T& normr0
    ) override {
        striped_warp_reduction_computeCG(A, b_d, x_d, n_iters, normr, normr0);
    }
    
    void compute_MG(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
            striped_warp_reduction_computeMG(A, x_d, y_d);
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in striped warp reduction." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_warp_reduction_computeSymGS(A, x_d, y_d);
    }
    
    void compute_SPMV(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the striped warp reduction Implementation.");
    }

    void compute_WAXPBY(
        sparse_CSR_Matrix<T>& A, // we pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "ERROR computeWAXPBY requires a striped Matrix as input in the striped warp reduction Implementation" << std::endl;
    }

    void compute_WAXPBY(
        striped_Matrix<T>& A, // we pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
            striped_warp_reduction_computeWAXPBY(A, x_d, y_d, w_d, alpha, beta);
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the sparse_CSR Matrix in striped warp reduction." << std::endl;
    }

    void compute_Dot(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        striped_warp_reduction_computeDot(A, x_d, y_d, result_d);
    }


    // Striped matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        striped_warp_reduction_computeSPMV(A, x_d, y_d);
    }


private:

    void striped_warp_reduction_computeCG(
        striped_Matrix<T> & A,
        T * b_d, T * x_d,
        int & n_iters, T& normr, T& normr0
    );

    void striped_warp_reduction_computeMG(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );

    void striped_warp_reduction_computeSPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );

    void striped_warp_reduction_computeDot(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d
    );

    void striped_warp_reduction_computeSymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );

    void striped_warp_reduction_computeWAXPBY(
        striped_Matrix<T> & A,
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
    );
};

// we expose the kernel in case we need to call it from another method
__global__ void striped_warp_reduction_SPMV_kernel(
        double* striped_A,
        int num_rows, int num_strips, int * j_min_i,
        double* x, double* y
    );
__global__ void striped_warp_reduction_dot_kernel(
    int num_rows,
    double * x_d,
    double * y_d,
    double * result_d
);


#endif // STRIPED_WARP_REDUCTION_CUH