# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

add_library(MatrixLib)

target_sources(MatrixLib PRIVATE
    sparse_CSR_Matrix.cpp
)

target_link_libraries(MatrixLib
    HPCGLib
    # CUDA::cusparse
)