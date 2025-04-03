#ifndef TESTING_HPP
#define TESTING_HPP

#include "MatrixLib/generations_hipified.hpp"
#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"
#include "MatrixLib/striped_Matrix_hipified.hpp"
#include "MatrixLib/striped_partial_Matrix_hipified.hpp"

#include "HPCGLib_hipified.hpp"
#include "HPCG_versions/cusparse_hipified.hpp"
#include "HPCG_versions/naiveStriped_hipified.cuh"
#include "HPCG_versions/striped_shared_mem_hipified.cuh"
#include "HPCG_versions/striped_warp_reduction_hipified.cuh"
#include "HPCG_versions/striped_preprocessed_hipified.cuh"
#include "HPCG_versions/striped_coloring_hipified.cuh"
#include "HPCG_versions/no_store_striped_coloring_hipified.cuh"
#include "HPCG_versions/striped_coloringPrecomputed_hipified.cuh"
#include "HPCG_versions/striped_box_coloring_hipified.cuh"
#include "HPCG_versions/striped_COR_box_coloring_hipified.cuh"

#include "HPCG_versions/striped_multi_GPU_hipified.cuh"

#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/utils_hipified.hpp"
#include "UtilLib/utils_hipified.cuh"

#include <iostream>

#define RANDOM_SEED 42
#define VERIFY_CG_WITH_PRECONDITIONING 0
#define HPCG_OUTPUT_TEST_FOLDER "../../../hpcg_output"

bool run_all_util_tests(int nx, int ny, int nz);

// MatrixLib testing functions
bool run_all_matrixLib_tests(int nx, int ny, int nz);

bool read_save_test(sparse_CSR_Matrix<DataType>& A, std::string info);
bool read_save_test(striped_Matrix<DataType>& A, std::string info);


// abstract test functions from HPCG_functions
bool test_CG(
    HPCG_functions<DataType>& implementation
);

bool test_MG(
    HPCG_functions<DataType>& implementation
);

// bool test_CG(
//     HPCG_functions<DataType>& uut,
//     striped_Matrix<DataType> & A,
//     DataType * x_d, DataType * y_d
// );

bool test_SPMV(
    HPCG_functions<DataType>& baseline, HPCG_functions<DataType>& uut,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d // the vectors x is already on the device
    );

// this one supports testing a striped matrix i.e. the naiveStriped implementation
bool test_SPMV(
    HPCG_functions<DataType>& baseline, HPCG_functions<DataType>& uut,
    striped_Matrix<DataType> & A,        
    DataType * x_d // the vectors x is already on the device
        
);

bool test_Dot(
    HPCG_functions<DataType>& baseline, HPCG_functions<DataType>& uut,
    striped_Matrix<DataType> & A, // we pass A for the metadata
    DataType * x_d, DataType * y_d
);

bool test_Dot(
    HPCG_functions<DataType>& uut,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d // the vectors x, y and result are already on the device
);

bool test_Dot(
    HPCG_functions<DataType>&uut,
    int nx, int ny, int nz
);

bool test_WAXPBY(
    HPCG_functions<DataType>& uut,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
);

bool test_WAXPBY(
    HPCG_functions<DataType>& uut,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d,
    DataType a, DataType b
);

bool test_SymGS(
    HPCG_functions<DataType>& uut,
    sparse_CSR_Matrix<DataType> & A
);
bool test_SymGS(
    HPCG_functions<DataType>& uut, HPCG_functions<DataType>& baseline,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d // the vectors x and y are already on the device
);

bool test_SymGS(
    HPCG_functions<DataType>& baseline, HPCG_functions<DataType>& uut,
    striped_Matrix<DataType> & striped_A,
        
    DataType * y_d // the vectors x is already on the device
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
bool run_stripedColoringPrecomputed_filebased_tests();
bool run_stripedColoringPrecomputed_tests(int nx, int ny, int nz);
bool run_stripedBoxColoring_tests(int nx, int ny, int nz);
bool run_no_store_stripedColoring_tests(int nx, int ny, int nz);
bool run_no_store_stripedColoring_filebased_tests();
bool run_COR_BoxColoring_tests(int nx, int ny, int nz);

#endif // TESTING_HPP