#include "testing.hpp"

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveBanded.cuh"

#include "cuda_utils.hpp"

#include <iostream>

// these functions are really just wrappers to check for correctness
// hence the only thing the function needs to allcoate is space for the outputs
// depending on the versions they may require different inputs, hence the method overloading

// in this case both versions require the same inputs 
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    double * x_d // the vectors x is already on the device
        
){  

    int num_rows = A.get_num_rows();
    std::vector<double> y_baseline(num_rows, 0.0);
    std::vector<double> y_uut(num_rows, 0.0);

    double * y_baseline_d;
    double * y_uut_d;

    CHECK_CUDA(cudaMalloc(&y_baseline_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_uut_d, num_rows * sizeof(double)));

    baseline.compute_SPMV(A,
                          A_row_ptr_d, A_col_idx_d, A_values_d,
                          x_d, y_baseline_d);

    uut.compute_SPMV(A,
                    A_row_ptr_d, A_col_idx_d, A_values_d,
                    x_d, y_uut_d);

    // and now we need to copy the result back and de-allocate the memory
    CHECK_CUDA(cudaMemcpy(y_baseline.data(), y_baseline_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y_uut.data(), y_uut_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(y_baseline_d));
    CHECK_CUDA(cudaFree(y_uut_d));

    // compare the results
    for (int i = 0; i < num_rows; i++) {
        if (y_baseline[i] != y_uut[i]) {
            std::cerr << "ERROR: cuSparse and Naive Banded SPMV results do not match." << std::endl;
            
            return false;
        }
    }
    return true;
}

// in this case the baseline requires CSR and the UUT requires both CSR and banded
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    
    double * banded_A_d, // the matrix A is already on the device
    int num_rows, int num_cols, // these refer to the shape of the banded matrix
    int num_bands, // the number of bands in the banded matrix
    int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        
    double * x_d // the vectors x is already on the device
        
){

    int num_rows_baseline = A.get_num_rows();
    std::vector<double> y_baseline(num_rows, 0.0);
    std::vector<double> y_uut(num_rows, 0.0);

    double * y_baseline_d;
    double * y_uut_d;

    CHECK_CUDA(cudaMalloc(&y_baseline_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_uut_d, num_rows * sizeof(double)));

    baseline.compute_SPMV(A,
                          A_row_ptr_d, A_col_idx_d, A_values_d,
                          x_d, y_baseline_d);

    uut.compute_SPMV(A,
                    banded_A_d,
                    num_rows, num_cols,
                    num_bands,
                    j_min_i_d,
                    x_d, y_uut_d);

    // and now we need to copy the result back and de-allocate the memory
    CHECK_CUDA(cudaMemcpy(y_baseline.data(), y_baseline_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y_uut.data(), y_uut_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(y_baseline_d));
    CHECK_CUDA(cudaFree(y_uut_d));

    // compare the results
    for (int i = 0; i < num_rows; i++) {
        if (y_baseline[i] != y_uut[i]) {
            std::cerr << "Error: cuSparse and Naive Banded SPMV results do not match." << std::endl;
            
            return false;
        }
    }
    return true;
}