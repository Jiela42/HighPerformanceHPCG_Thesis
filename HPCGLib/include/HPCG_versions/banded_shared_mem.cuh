#ifndef BANDEDSHAREDMEMORY_CUH
#define BANDEDSHAREDMEMORY_CUH

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
class Banded_Shared_Memory_Implementation : public HPCG_functions<T> {
public:

    std::string version_name = "Banded explicit Shared Memory";
    std::string additional_parameters = "num_threads = 1024, num_blocks = theoretical maximum";

    void compute_CG(sparse_CSR_Matrix<T>& A, std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in Banded Shared Memory." << std::endl;
    }
    
    void compute_MG(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in Banded Shared Memory." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in Banded Shared Memory." << std::endl;
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a banded Matrix as input in the Banded Shared Memory Implementation.");
    }

    void compute_WAXPBY(
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Banded Shared Memory." << std::endl;
    }

    void compute_Dot(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented in Banded Shared Memory." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented in Banded Shared Memory and needs a banded matrix, not a CSR matrix." << std::endl;
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
        )  {
        banded_shared_memory_computeSPMV(A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, x_d, y_d);
    }

private:

    void banded_shared_memory_computeSPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// kernel functions, because they cannot be member functions
__global__ void banded_shared_memory_SPMV_kernel(
    int rows_per_sm, int num_x_elem, int num_consecutive_memory_regions,
    int* min_j, int* max_j,
    double* banded_A,
    int num_rows, int num_bands, int * j_min_i,
    double* x, double* y
);

#endif // BANDEDSHAREDMEMORY_CUH