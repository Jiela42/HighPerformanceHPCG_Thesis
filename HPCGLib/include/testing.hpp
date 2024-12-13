#ifndef TESTING_HPP
#define TESTING_HPP

#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveStriped.cuh"
#include "HPCG_versions/striped_shared_mem.cuh"
#include "HPCG_versions/striped_warp_reduction.cuh"
#include "HPCG_versions/striped_preprocessed.cuh"
#include "HPCG_versions/striped_coloring.cuh"
#include "HPCG_versions/no_store_striped_coloring.cuh"


#include "UtilLib/cuda_utils.hpp"

#include <iostream>

#define RANDOM_SEED 42

// MatrixLib testing functions
bool run_all_matrixLib_tests(int nx, int ny, int nz);

bool read_save_test(sparse_CSR_Matrix<double> A, std::string info);
bool read_save_test(striped_Matrix<double> A, std::string info);


// abstract test functions from HPCG_functions
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    double * x_d // the vectors x is already on the device
    );

// this one supports testing a striped matrix i.e. the naiveStriped implementation
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    
    double * striped_A_d, // the matrix A is already on the device
    int num_rows, int num_cols, // these refer to the shape of the striped matrix
    int num_stripes, // the number of stripes in the striped matrix
    int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        
    double * x_d // the vectors x is already on the device
        
);

bool test_Dot(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & A, // we pass A for the metadata
    double * x_d, double * y_d
);

bool test_Dot(
    HPCG_functions<double>&uut,
    int nx, int ny, int nz
);

bool test_SymGS(
    HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A
);
bool test_SymGS(
    HPCG_functions<double>& uut, HPCG_functions<double>& baseline,
    sparse_CSR_Matrix<double> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the matrix A is already on the device
    double * x_d, double * y_d // the vectors x and y are already on the device
);

bool test_SymGS(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & striped_A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d, // the CSR matrix A is already on the device
    
    double * striped_A_d, // the striped matrix A is already on the device
    int num_rows, int num_cols, // these refer to the shape of the striped matrix
    int num_stripes, // the number of stripes in the striped matrix
    int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        
    double * y_d // the vectors x is already on the device
);

// functions that call the abstract tests in order to test full versions
bool run_cuSparse_tests(int nx, int ny, int nz);
bool run_naiveStriped_tests(int nx, int ny, int nz);
bool run_stripedSharedMem_tests(int nx, int ny, int nz);
bool run_stripedWarpReduction_tests(int nx, int ny, int nz);
bool run_stripedPreprocessed_tests(int nx, int ny, int nz);
bool run_stripedColored_tests(int nx, int ny, int nz);

#endif // TESTING_HPP