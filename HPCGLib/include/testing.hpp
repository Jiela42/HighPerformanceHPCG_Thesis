#ifndef TESTING_HPP
#define TESTING_HPP

#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/banded_Matrix.hpp"

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveBanded.cuh"

#include "cuda_utils.hpp"

#include <iostream>

// MatrixLib testing functions
void run_all_tests(int nx, int ny, int nz);

void read_save_test(sparse_CSR_Matrix<double> A, std::string info);
void read_save_test(banded_Matrix<double> A, std::string info);


// abstract test functions from HPCG_functions
void test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    const sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    double * x_d // the vectors x is already on the device
    );

// functions that call the abstract tests in order to test full versions
void run_naiveBanded_tests(int nx, int ny, int nz);

#endif // TESTING_HPP