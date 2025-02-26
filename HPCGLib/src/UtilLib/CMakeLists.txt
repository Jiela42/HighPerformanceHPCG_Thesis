# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

add_library(UtilLib)

find_package(CUDAToolkit REQUIRED)
# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# specify cuda standard
# set(CMAKE_CUDA_STANDARD 12)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Set CUDA architectures
# set(CMAKE_CUDA_ARCHITECTURES 70 86)

enable_language(CUDA)

# Set compile options for CUDA
target_compile_options(HPCGLib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    # --std=c++12
    # -arch=sm_86
>)

target_sources(UtilLib PRIVATE
utils.cu
utils.cpp
)

target_link_libraries(UtilLib
    HPCGLib
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
)