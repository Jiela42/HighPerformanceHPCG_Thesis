#ifndef STRIPED_WARP_REDUCTION_CUH
#define STRIPED_WARP_REDUCTION_CUH

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
class striped_warp_reduction_Implementation : public HPCG_functions<T> {
public:

    striped_warp_reduction_Implementation(){
        // overwritting the inherited variables

        this->version_name = "Striped Warp Reduction";
        this->implementation_type = Implementation_Type::STRIPED;
        this->SPMV_implemented = true;
        this->Dot_implemented = true;
        this->SymGS_implemented = true;

    }

    void compute_CG(sparse_CSR_Matrix<T>& A, std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in striped warp reduction." << std::endl;
    }
    
    void compute_MG(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in striped warp reduction." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in striped warp reduction." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * striped_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_strips, // the number of strips in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_warp_reduction_computeSymGS(A, striped_A_d, num_rows, num_cols, num_strips, j_min_i, x_d, y_d);
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the striped warp reduction Implementation.");
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in striped warp reduction." << std::endl;
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
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * striped_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the striped matrix
        int num_strips, // the number of strips in the striped matrix
        int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        striped_warp_reduction_computeSPMV(A, striped_A_d, num_rows, num_cols, num_strips, j_min_i_d, x_d, y_d);
    }

private:

    void striped_warp_reduction_computeSPMV(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * striped_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the striped matrix
        int num_strips, // the number of strips in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );

    void striped_warp_reduction_computeDot(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d
    );

    void striped_warp_reduction_computeSymGS(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * striped_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_strips, // the number of strips in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
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