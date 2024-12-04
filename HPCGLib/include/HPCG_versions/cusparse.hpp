#ifndef CUSPARSE_HPP
#define CUSPARSE_HPP

#include "HPCGLib.hpp"
#include "HPCG_versions/banded_warp_reduction.cuh"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <iostream>
#include <string>

template <typename T>
class cuSparse_Implementation : public HPCG_functions<T> {
public:

    std::string version_name = "CSR-Implementation";
    Implementation_Type implementation_type = Implementation_Type::CSR;


    void compute_CG(sparse_CSR_Matrix<T>& A, std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in cuSparse_Implementation." << std::endl;
    }
    
    void compute_MG(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in cuSparse_Implementation." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        cusparse_computeSymGS(
            A,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            x_d, y_d);
    }

    void compute_SymGS(
        banded_Matrix<T> & A, // we pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) override{
        std::cerr << "Error: compute_SymGS needs different parameters for the cuSparse_Implementation." << std::endl;
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        cusparse_computeSPMV(
            A,
            A_row_ptr_d, A_col_idx_d, A_values_d,
            x_d, y_d);
    }

    void compute_SPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Error: compute_SPMV needs different parameters for the cuSparse_Implementation." << std::endl;
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in cuSparse_Implementation." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        cusparse_computeDot(
            A,
            x_d, y_d, result_d);
    }

    void compute_Dot(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d // the vectors x, y and result are already on the device
        ) override {
        warp_Implementation.compute_Dot(
            A,
            x_d, y_d, result_d);
    }
    

private:

    banded_warp_reduction_Implementation<T> warp_Implementation;

    // here come the cuSparse functions
    void cusparse_computeSPMV(
    sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
    T * x_d, T * y_d // the vectors x and y are already on the device
    );

    void cusparse_computeDot(
        sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d // the vectors x, y and result are already on the device
    );

    void cusparse_computeSymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

#endif // CUSPARSE_HPP