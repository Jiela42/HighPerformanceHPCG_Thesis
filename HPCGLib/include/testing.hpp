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
#include "HPCG_versions/striped_coloringPrecomputed.cuh"
#include "HPCG_versions/striped_box_coloring.cuh"


#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.hpp"
#include "UtilLib/utils.cuh"

#include <iostream>

#define RANDOM_SEED 42
#define VERIFY_CG_WITH_PRECONDITIONING 0

bool run_all_util_tests(int nx, int ny, int nz);

// MatrixLib testing functions
bool run_all_matrixLib_tests(int nx, int ny, int nz);

bool read_save_test(sparse_CSR_Matrix<double>& A, std::string info);
bool read_save_test(striped_Matrix<double>& A, std::string info);


// abstract test functions from HPCG_functions
bool test_CG(
    HPCG_functions<double>& implementation,
    std::string test_folder
);

bool test_MG(
    HPCG_functions<double>& implementation,
    std::string test_folder
);

// bool test_CG(
//     HPCG_functions<double>& uut,
//     striped_Matrix<double> & A,
//     double * x_d, double * y_d
// );

bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A,
    double * x_d // the vectors x is already on the device
    );

// this one supports testing a striped matrix i.e. the naiveStriped implementation
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & A,        
    double * x_d // the vectors x is already on the device
        
);

bool test_Dot(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & A, // we pass A for the metadata
    double * x_d, double * y_d
);

bool test_Dot(
    HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d // the vectors x, y and result are already on the device
);

bool test_Dot(
    HPCG_functions<double>&uut,
    int nx, int ny, int nz
);

bool test_WAXPBY(
    HPCG_functions<double>& uut,
    striped_Matrix<double> & A,
    double * x_d, double * y_d
);

bool test_WAXPBY(
    HPCG_functions<double>& uut,
    striped_Matrix<double> & A,
    double * x_d, double * y_d,
    double a, double b
);

bool test_SymGS(
    HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A
);
bool test_SymGS(
    HPCG_functions<double>& uut, HPCG_functions<double>& baseline,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d // the vectors x and y are already on the device
);

bool test_SymGS(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & striped_A,
        
    double * y_d // the vectors x is already on the device
);

// functions that call the abstract tests in order to test full versions
bool run_cuSparse_tests(int nx, int ny, int nz);
bool run_amgx_tests(int nx, int ny, int nz);
bool run_naiveStriped_tests(int nx, int ny, int nz);
bool run_stripedSharedMem_tests(int nx, int ny, int nz);
bool run_stripedWarpReduction_tests(int nx, int ny, int nz);
bool run_stripedWarpReduction_filebased_tests();
bool run_stripedPreprocessed_tests(int nx, int ny, int nz);
bool run_stripedColored_tests(int nx, int ny, int nz);
bool run_stripedColoringPrecomputed_tests(int nx, int ny, int nz);
bool run_stripedBoxColoring_tests(int nx, int ny, int nz);

#endif // TESTING_HPP