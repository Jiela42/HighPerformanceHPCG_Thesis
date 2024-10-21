# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

add_library(HPCGLib)

find_package(CUDAToolkit REQUIRED)
# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# specify cuda standard
# set(CMAKE_CUDA_STANDARD 12)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set compile options for CUDA
target_compile_options(HPCGLib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    # --std=c++12
    -arch=sm_86
>)

# Set properties for CUDA compilation
# set_target_properties(HPCGLib PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )


target_sources(HPCGLib PRIVATE
    # HPCG_original/HPCG_original_test.cpp

    cusparse/cusparse_computeSPMV.cpp

    naiveBanded/naiveBanded_compute_SPMV.cu
    naiveBanded/naiveBanded_kernel_SPMV.cu
)

target_link_libraries(
    HPCGLib
    TimingLib
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
)
