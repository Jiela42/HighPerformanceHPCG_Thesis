# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

add_library(HPCGLib)

target_sources(HPCGLib PRIVATE
    HPCG_original/HPCG_original_test.cpp

    cusparse/cusparse_computeSPMV.cpp
    cusparse/cusparse_kernel_SYMGS.cu
)

target_link_libraries(
    HPCGLib
    TimingLib
    cusparse
    # CUDA::cusparse
)