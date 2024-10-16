#ifndef NAIVEBANDED_HPP
#define NAIVEBANDED_HPP
#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <iostream>
#include <string>

template <typename T>
class naiveBanded_Implementation : public HPCG_functions<T> {
public:

    std::string version_name = "Naive Banded";

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
        std::cerr << "Warning: compute_SPMV is not implemented in Naive Banded." << std::endl;
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
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) {
        naiveBanded_compute_SPMV(A, banded_A_d, num_rows, num_cols, j_min_i, x_d, y_d);
    }

private:
    void naiveBanded_compute_SPMV(
        const sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
   
};

#endif // NAIVEBANDED_HPP