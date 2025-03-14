cmake_minimum_required(VERSION 3.10)

project(Testing LANGUAGES CXX CUDA)

# Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and make sure it's available
find_package(MPI REQUIRED)

add_library(TestHPCGLib
    HPCG_versions_tests/cuSparse_tests.cpp
    HPCG_versions_tests/naiveStriped_tests.cpp
    HPCG_versions_tests/stripedSharedMem_tests.cpp
    HPCG_versions_tests/stripedWarpReduction_tests.cpp
    HPCG_versions_tests/stripedPreprocessed_tests.cpp
    HPCG_versions_tests/stripedColored_tests.cpp
    HPCG_versions_tests/stripedColoringPrecomputed_tests.cpp
    HPCG_versions_tests/stripedBoxColoring_tests.cpp
    HPCG_versions_tests/stripedNoStore_tests.cpp
    
    UtilLib_tests.cpp
    MatrixLib_tests.cpp
    HPCG_versions_tests/test_calls.cpp
)

add_executable(run_coloring
    run_coloring.cpp
)

add_executable(run_AllTests
    run_all_tests.cpp
)

add_executable(run_get_Norm
    run_get_Norm.cpp
)

add_executable(run_example
    run_example.cpp
)

add_executable(run_multi_GPU
    run_multi_GPU.cpp
)

target_link_libraries(TestHPCGLib
    HPCGLib
    MatrixLib
    UtilLib
)

target_link_libraries(run_AllTests
    HPCGLib
    MatrixLib
    TestHPCGLib
    UtilLib
    stdc++fs
)

target_link_libraries(run_coloring
    HPCGLib
    MatrixLib
)

target_link_libraries(run_get_Norm
    HPCGLib
    MatrixLib
    UtilLib
)

target_link_libraries(run_example
    HPCGLib
    MatrixLib
)

target_link_libraries(run_multi_GPU
    HPCGLib
    MatrixLib
    UtilLib
    MPI::MPI_CXX
)
