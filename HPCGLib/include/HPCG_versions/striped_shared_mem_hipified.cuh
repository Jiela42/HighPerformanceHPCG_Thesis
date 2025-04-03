#include "hip/hip_runtime.h"
#ifndef STRIPED_SHARED_MEM_CUH
#define STRIPED_SHARED_MEM_CUH

#include "HPCGLib_hipified.hpp"
#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"
#include "MatrixLib/striped_Matrix_hipified.hpp"
#include "UtilLib/cuda_utils_hipified.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <hip/hip_runtime.h>

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

    void compute_CG(
        striped_Matrix<T> & A,
        T * b_d, T * x_d,
        int & n_iters, T& normr, T& normr0) override {
        std::cerr << "Warning: compute_CG is not implemented in Striped Shared Memory." << std::endl;
    }
    
    void compute_MG(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) override {
        std::cerr << "Warning: compute_SymGS is not implemented in Striped Shared Memory." << std::endl;
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the Striped Shared Memory Implementation.");
    }

    void compute_WAXPBY(
        sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in Striped Shared Memory." << std::endl;
    }
    void compute_WAXPBY(
        striped_Matrix<T>& A, //we only pass A for the metadata
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
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        )  {
        striped_shared_memory_computeSPMV(A, x_d, y_d);
    }

private:

    void striped_shared_memory_computeSPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernels in case we need them in other methods
__global__ void striped_shared_memory_SPMV_kernel(
        local_int_t rows_per_sm, local_int_t num_x_elem, local_int_t num_consecutive_memory_regions,
        local_int_t* min_j, local_int_t* max_j,
        DataType* striped_A,
        local_int_t num_rows, int num_stripes, local_int_t * j_min_i,
        DataType* x, DataType* y
    );
#endif // STRIPED_SHARED_MEM_CUH