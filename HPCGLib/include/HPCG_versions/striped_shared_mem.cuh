#ifndef STRIPED_SHARED_MEM_CUH
#define STRIPED_SHARED_MEM_CUH

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
class Striped_Shared_Memory_Implementation : public HPCG_functions<T> {
public:

    Striped_Shared_Memory_Implementation(){
        // overwritting the inherited variables

        this->version_name = "Striped Shared Memory";
        this->additional_parameters = "num_threads = 1024, num_blocks = theoretical maximum";
        
        this->implementation_type = Implementation_Type::STRIPED;
        this->SPMV_implemented = true;
    }

    void compute_CG(sparse_CSR_Matrix<T>& A, std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in Striped Shared Memory." << std::endl;
    }
    
    void compute_MG(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * striped_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols,
        int num_stripes, // the number of stripes in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the Striped Shared Memory Implementation.");
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_Dot(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented in Striped Shared Memory and needs a striped matrix, not a CSR matrix." << std::endl;
    }



    // Striped matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * striped_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the striped matrix
        int num_stripes, // the number of stripes in the striped matrix
        int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
        )  {
        striped_shared_memory_computeSPMV(A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, x_d, y_d);
    }

private:

    void striped_shared_memory_computeSPMV(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * striped_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the striped matrix
        int num_stripes, // the number of stripes in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernels in case we need them in other methods
__global__ void striped_shared_memory_SPMV_kernel(
        int rows_per_sm, int num_x_elem, int num_consecutive_memory_regions,
        int* min_j, int* max_j,
        double* striped_A,
        int num_rows, int num_stripes, int * j_min_i,
        double* x, double* y
    );
#endif // STRIPED_SHARED_MEM_CUH