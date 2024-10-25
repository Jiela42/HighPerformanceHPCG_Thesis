#ifndef TESTING_HPP
#define TESTING_HPP

#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/banded_Matrix.hpp"

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveBanded.cuh"

#include "UtilLib/cuda_utils.hpp"

#include <iostream>

// MatrixLib testing functions
bool run_all_matrixLib_tests(int nx, int ny, int nz);

bool read_save_test(sparse_CSR_Matrix<double> A, std::string info);
bool read_save_test(banded_Matrix<double> A, std::string info);


// abstract test functions from HPCG_functions
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    double * x_d // the vectors x is already on the device
    );

// this one supports testing a banded matrix i.e. the naiveBanded implementation
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    
    double * banded_A_d, // the matrix A is already on the device
    int num_rows, int num_cols, // these refer to the shape of the banded matrix
    int num_bands, // the number of bands in the banded matrix
    int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        
    double * x_d // the vectors x is already on the device
        
);

// functions that call the abstract tests in order to test full versions
bool run_naiveBanded_tests(int nx, int ny, int nz);

#endif // TESTING_HPP