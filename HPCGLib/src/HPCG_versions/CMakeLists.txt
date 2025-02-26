# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

project(HPCGLib LANGUAGES CXX CUDA)

add_library(HPCGLib)

find_package(CUDAToolkit REQUIRED)
# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# specify cuda standard
# set(CMAKE_CUDA_STANDARD 12)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70 86)

# Set compile options for CUDA
target_compile_options(HPCGLib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    # --std=c++12
    # --std=c++11
    # -arch=sm_86
>)

target_sources(HPCGLib PRIVATE

    cusparse/cusparse_computeSPMV.cpp
    cusparse/cusparse_computeDot.cu
    cusparse/cusparse_computeSymGS.cu

    naiveStriped/naiveStriped_computeSPMV.cu

    striped_shared_mem/striped_shared_mem_computeSPMV.cu

    striped_warp_reduction/striped_warp_reduction_computeSPMV.cu
    striped_warp_reduction/striped_warp_reduction_computeSymGS.cu
    striped_warp_reduction/striped_warp_reduction_computeDot.cu
    striped_warp_reduction/striped_warp_reduction_computeWAXPBY.cu
    striped_warp_reduction/striped_warp_reduction_computeCG.cu
    striped_warp_reduction/striped_warp_reduction_computeMG.cu

    striped_preprocessed/striped_preprocessed_computeSymGS.cu

    striped_coloring/striped_coloring_computeSymGS.cu

    no_store_striped_coloring/no_store_striped_coloring_computeSymGS.cu

    striped_coloringPrecomputed/striped_coloringPrecomputed_computeSymGS.cu

    striped_box_coloring/striped_box_coloring_computeSymGS.cu

)

target_link_libraries(
    HPCGLib
    TimingLib
    UtilLib
    MatrixLib
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
    # ${AMGX_LIB}
    # Thrust
)
