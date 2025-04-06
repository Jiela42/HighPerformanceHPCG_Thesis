# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name
project(HPCGlib)

# Set CUDA architecture to 90a
# set(CMAKE_CUDA_ARCHITECTURES 90a 86)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a")
include_directories("/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nccl-2.20.3-1-l2d55ru4lupkbzxnqm2b5nrvlgnt3v3s/include")

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(benchmarking)
