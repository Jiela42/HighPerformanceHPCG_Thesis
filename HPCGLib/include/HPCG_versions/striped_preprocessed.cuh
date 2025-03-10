#ifndef STRIPED_PREPROCESSED_CUH
#define STRIPED_PREPROCESSED_CUH

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
class striped_preprocessed_Implementation : public HPCG_functions<T> {
public:

    striped_preprocessed_Implementation(){

        // overwritting the inherited variables

        this->version_name = "Striped Preprocessed";
        this->additional_parameters = "SymGS cooperation number = 16";

        this->implementation_type = Implementation_Type::STRIPED;

        this->SymGS_implemented = true;
        
    }

    void compute_CG(
        striped_Matrix<T> & A,
        T * b_d, T * x_d,
        int & n_iters, T& normr, T& normr0
    ) override {
        std::cerr << "Warning: compute_CG is not implemented in Striped Preprocessed." << std::endl;
    }
    
    void compute_MG(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in Striped Preprocessed." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in Striped Preprocessed." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_preprocessed_computeSymGS(A, x_d, y_d);
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the Striped Preprocessed Implementation.");
    }

    void compute_WAXPBY(
        sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Striped Preprocessed." << std::endl;
    }
    void compute_WAXPBY(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the sparse_CSR Matrix in Striped Preprocessed." << std::endl;
    }

    void compute_Dot(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the striped Matrix in Striped Preprocessed." << std::endl;
    }


    // Striped matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SPMV requires different arguments in Striped Preprocessed." << std::endl;
    }

private:

    void striped_preprocessed_computeSymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method


#endif // STRIPED_PREPROCESSED_CUH