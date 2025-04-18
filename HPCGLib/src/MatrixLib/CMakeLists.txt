# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

project(MatrixLib LANGUAGES CXX CUDA)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MatrixLib)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
# set(CMAKE_CUDA_ARCHITECTURES 86 70)
target_compile_options(MatrixLib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    # --std=c++12
    # -arch=sm_86
>)


target_sources(MatrixLib PRIVATE
    sparse_CSR_Matrix.cpp
    striped_partial_Matrix.cpp
    striped_Matrix.cpp
    generations.cpp
    generations.cu
    coloring.cu
)

target_link_libraries(MatrixLib
    HPCGLib
    CUDA::cudart
)