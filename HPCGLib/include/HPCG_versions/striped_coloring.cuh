#ifndef STRIPED_COLORING_CUH
#define STRIPED_COLORING_CUH

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
class striped_coloring_Implementation : public HPCG_functions<T> {
public:

    striped_coloring_Implementation(){

        // overwritting the inherited variables

        this->version_name = "Striped coloring (pre-computing COR Format)";
        // this->additional_parameters = "SymGS cooperation number = 16";

        this->implementation_type = Implementation_Type::STRIPED;

        this->SymGS_implemented = true;
        
    }

    void compute_CG(sparse_CSR_Matrix<T>& A, std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in Striped coloring." << std::endl;
    }
    
    void compute_MG(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in Striped coloring." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in Striped coloring." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * striped_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_bands, // the number of bands in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_coloring_computeSymGS(A, striped_A_d, num_rows, num_cols, num_bands, j_min_i, x_d, y_d);
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the Striped coloring Implementation.");
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Striped coloring." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the sparse_CSR Matrix in Striped coloring." << std::endl;
    }

    void compute_Dot(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the striped Matrix in Striped coloring." << std::endl;
    }


    // Striped matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * striped_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the striped matrix
        int num_bands, // the number of bands in the striped matrix
        int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SPMV requires different arguments in Striped coloring." << std::endl;
    }

private:

    void striped_coloring_computeSymGS(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * striped_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_bands, // the number of bands in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method
__global__ void striped_coloring_half_SymGS_kernel(
    int color, int * color_pointer, int * color_sorted_rows,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    double * x, double * y
);

#endif // STRIPED_COLORING_CUH