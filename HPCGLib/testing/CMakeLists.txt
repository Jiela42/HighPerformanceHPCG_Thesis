

# Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and make sure it's available

file(GLOB SOURCES "*.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_library(TestHPCGLib
    ${SOURCES}
)


add_executable(run_coloring
    run_coloring_hipified.cpp
)

add_executable(run_AllTests
    run_all_tests_hipified.cpp
)

add_executable(run_get_Norm
    run_get_Norm_hipified.cpp
)

add_executable(run_example
    run_example_hipified.cpp
)

add_executable(run_multi_GPU_tests
    run_multi_GPU_tests_hipified.cpp
)

add_executable(run_multi_GPU
    run_multi_GPU_hipified.cpp
)

target_link_libraries(TestHPCGLib PUBLIC
    HPCGLib
    MatrixLib
    UtilLib
)

target_link_libraries(run_AllTests PUBLIC
    HPCGLib
    MatrixLib
    TestHPCGLib
    UtilLib
    stdc++fs
)

target_link_libraries(run_coloring PUBLIC
    HPCGLib
    MatrixLib
)

target_link_libraries(run_get_Norm PUBLIC
    HPCGLib
    MatrixLib
    UtilLib
)

target_link_libraries(run_example PUBLIC
    HPCGLib
    MatrixLib
)

target_link_libraries(run_multi_GPU PUBLIC
    HPCGLib
    MatrixLib
    UtilLib
    hipsparse
    hipblas
    MPI::MPI_CXX
)

target_link_libraries(run_multi_GPU_tests PUBLIC
    HPCGLib
    MatrixLib
    UtilLib
    MPI::MPI_CXX
    hipsparse
    hipblas
    rccl
)