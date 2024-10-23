#ifndef NAIVEBANDED_CUH
#define NAIVEBANDED_CUH

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/banded_Matrix.hpp"

#include <vector>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include "cuda_utils.hpp"

template <typename T>
class naiveBanded_Implementation : public HPCG_functions<T> {
public:

    std::string version_name = "Naive Banded";

    // naiveBanded_Implementation() {
    //     std::cerr << "Warning: Naive Banded is created." << std::endl;
    // }
    // ~naiveBanded_Implementation() {
    //     std::cerr << "Warning: Naive Banded is being destroyed." << std::endl;
    // }

    void compute_CG(const sparse_CSR_Matrix<T>& A, const std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in Naive Banded." << std::endl;
    }
    
    void compute_MG(
        const sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in Naive Banded." << std::endl;
    }

    void compute_SymGS(
        const sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in Naive Banded." << std::endl;
    }

    void compute_SPMV(
        const sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        // This is relevant for the tests, but will not be used
        // In this function we just rewrite A to match the banded format
        banded_Matrix<double> banded_A;
        banded_A.banded_3D27P_Matrix_from_CSR(A);
        int num_bands = banded_A.get_num_bands();

        double * banded_A_d;
        int * y_min_i_d;

        int num_rows = A.get_num_rows();
        int num_cols = A.get_num_cols();

        CHECK_CUDA(cudaMalloc(&banded_A_d, num_bands * num_rows * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&y_min_i_d, num_rows * sizeof(double)));
        
        CHECK_CUDA(cudaMemcpy(banded_A_d, banded_A.get_values().data(), num_bands * num_rows * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(y_min_i_d, banded_A.get_j_min_i().data(), num_rows * sizeof(int), cudaMemcpyHostToDevice));
        
        // now just call the implementation
        compute_SPMV(A, banded_A_d, num_rows, num_cols, num_bands, y_min_i_d, x_d, y_d);
    
        // anything I initialized on the device, I need to free again
        CHECK_CUDA(cudaFree(banded_A_d));
        CHECK_CUDA(cudaFree(y_min_i_d));

    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Naive Banded." << std::endl;
    }

    void compute_Dot(
        T * x_d, T & y_d, T & result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented in Naive Banded." << std::endl;
    }


    // Banded matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        const sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) {
        naiveBanded_computeSPMV(A, banded_A_d, num_rows, num_cols, num_bands, j_min_i, x_d, y_d);
    }

private:

    void naiveBanded_computeSPMV(
        const sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// kernel functions, because they cannot be member functions
__global__ void naiveBanded_SPMV_kernel(
double* banded_A,
int num_rows, int num_bands, int * j_min_i,
double* x, double* y
);

#endif // NAIVEBANDED_CUH