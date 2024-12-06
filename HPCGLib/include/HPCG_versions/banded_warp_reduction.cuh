#ifndef BANDED_WARP_REDUCTION_CUH
#define BANDED_WARP_REDUCTION_CUH

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/banded_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
class banded_warp_reduction_Implementation : public HPCG_functions<T> {
public:

    std::string version_name = "Banded Warp Reduction (loop body in method)";
    Implementation_Type implementation_type = Implementation_Type::BANDED;


    void compute_CG(sparse_CSR_Matrix<T>& A, std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in banded warp reduction." << std::endl;
    }
    
    void compute_MG(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in banded warp reduction." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in banded warp reduction." << std::endl;
    }

    void compute_SymGS(
        banded_Matrix<T> & A, // we pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        banded_warp_reduction_computeSymGS(A, banded_A_d, num_rows, num_cols, num_bands, j_min_i, x_d, y_d);
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a banded Matrix as input in the banded warp reduction Implementation.");
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in banded warp reduction." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the sparse_CSR Matrix in banded warp reduction." << std::endl;
    }

    void compute_Dot(
        banded_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        banded_warp_reduction_computeDot(A, x_d, y_d, result_d);
    }


    // Banded matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        banded_warp_reduction_computeSPMV(A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, x_d, y_d);
    }

private:

    void banded_warp_reduction_computeSPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );

    void banded_warp_reduction_computeDot(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d
    );

    void banded_warp_reduction_computeSymGS(
        banded_Matrix<T> & A, // we pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method
__global__ void banded_warp_reduction_SPMV_kernel(
        double* banded_A,
        int num_rows, int num_bands, int * j_min_i,
        double* x, double* y
    );
__global__ void banded_warp_reduction_dot_kernel(
    int num_rows,
    double * x_d,
    double * y_d,
    double * result_d
);


#endif // BANDED_WARP_REDUCTION_CUH