#ifndef HPCGLIB_HPP
#define HPCGLIB_HPP

#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <string>

// define number of iterations we want to have
#define num_bench_iter 10

template <typename T>
class HPCG_functions {
    public:
        const std::string version_name;
    // CG starts with having the data on the CPU
        virtual void compute_CG(sparse_CSR_Matrix<T> & A, std::vector<T> & b, std::vector<T> & x) = 0;
        
    // MG, SymGS, SPMV, WAXPBY and Dot have the data on the GPU already
        virtual void compute_MG(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_SymGS(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        // this version supports CSR
        virtual void compute_SPMV(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;
        
        // this version is for the banded matrix
        virtual void compute_SPMV(
            sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
            T * banded_A_d, // the matrix A is already on the device
            int num_rows, int num_cols, // these refer to the shape of the banded matrix
            int num_bands, // the number of bands in the banded matrix
            int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the banded matrix
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_WAXPBY(
            T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
            T alpha, T beta
            ) = 0;

        virtual void compute_Dot(
            T * x_d, T & y_d, T & result_d // again: the vectors x, y and result are already on the device
        ) = 0;

        bool test_before_bench = true;

        int getNumberOfIterations() const {
            return num_bench_iter;
    }
};

#endif // HPCGLIB_HPP