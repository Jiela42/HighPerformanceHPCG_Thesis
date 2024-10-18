#ifndef HPCG_ORIGINAL_HPP
#define HPCG_ORIGINAL_HPP

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <iostream>
#include <string>

template <typename T>
class HPCG_Original_Implementation : public HPCG_functions<T> {
public:

    std::string version_name = "HPCG_Original";

    void compute_CG(const sparse_CSR_Matrix<T>& A, const std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in HPCG_Original." << std::endl;
    }
    
    void compute_MG(
        const sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in HPCG_Original." << std::endl;
    }

    void compute_SymGS(
        const sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in HPCG_Original." << std::endl;
    }

    void compute_SPMV(
        const sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SPMV is not implemented in HPCG_Original." << std::endl;
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in HPCG_Original." << std::endl;
    }

    void compute_Dot(
        T * x_d, T & y_d, T & result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented in HPCG_Original." << std::endl;
    }

private:
   
};

#endif // HPCG_ORIGINAL_HPP