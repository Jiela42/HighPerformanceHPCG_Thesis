# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

project(HPCGLib LANGUAGES CXX CUDA)

add_library(HPCGLib)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${MPI_CXX_INCLUDE_DIRS})
include_directories(/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nccl-2.20.3-1-l2d55ru4lupkbzxnqm2b5nrvlgnt3v3s/include)


# specify cuda standard
# set(CMAKE_CUDA_STANDARD 12)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DMPI_GPU_AWARE)

# set(CMAKE_CUDA_ARCHITECTURES 86)

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

    multi_GPU_kernels/striped_box_coloring_multi_GPU_computeSymGS.cu
    multi_GPU_kernels/striped_warp_reduction_multi_GPU_computeCG.cu
    multi_GPU_kernels/striped_warp_reduction_multi_GPU_computeDot.cu
    multi_GPU_kernels/striped_warp_reduction_multi_GPU_computeMG.cu
    multi_GPU_kernels/striped_warp_reduction_multi_GPU_computeSPMV.cu
    multi_GPU_kernels/striped_warp_reduction_multi_GPU_computeWAXPBY.cu

    multi_GPU_halo_exchange/blocking_mpi_halo_exchange.cu
    multi_GPU_halo_exchange/non_blocking_mpi_halo_exchange.cu
    multi_GPU_halo_exchange/nccl_halo_exchange.cu

    striped_preprocessed/striped_preprocessed_computeSymGS.cu

    striped_coloring/striped_coloring_computeSymGS.cu

    no_store_striped_coloring/no_store_striped_coloring_computeSymGS.cu

    striped_coloringPrecomputed/striped_coloringPrecomputed_computeSymGS.cu

    striped_box_coloring/striped_box_coloring_computeSymGS.cu

    striped_COR_box_coloring/striped_box_coloring_computeSymGS.cu

)

target_link_libraries(
    HPCGLib
    TimingLib
    UtilLib
    MatrixLib
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
    MPI::MPI_CXX
)
