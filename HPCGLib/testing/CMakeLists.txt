cmake_minimum_required(VERSION 3.10)

project(Testing LANGUAGES CXX CUDA)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)

add_library(TestHPCGLib
    HPCG_versions_tests/cuSparse_tests.cpp
    HPCG_versions_tests/naiveBanded_tests.cpp
    HPCG_versions_tests/bandedSharedMem_tests.cpp
    HPCG_versions_tests/bandedWarpReduction_tests.cpp
    HPCG_versions_tests/bandedPreprocessed_tests.cpp
        
    MatrixLib_tests.cpp
    HPCG_versions_tests/test_calls.cpp
)

add_executable(run_coloring
    run_coloring.cpp
)

add_executable(run_AllTests
    run_all_tests.cpp
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
)

target_link_libraries(run_coloring
    HPCGLib
    MatrixLib
)