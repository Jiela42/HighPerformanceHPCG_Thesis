# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

add_library(HPCGLib)

find_package(CUDAToolkit REQUIRED)
# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

target_sources(HPCGLib PRIVATE
    # HPCG_original/HPCG_original_test.cpp

    cusparse/cusparse_computeSPMV.cpp

    naiveBanded/naiveBanded_compute_SPMV.cpp
)

target_link_libraries(
    HPCGLib
    TimingLib
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
)