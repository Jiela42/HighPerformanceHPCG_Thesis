# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name
project(HPCGlib)

# Set CUDA architecture to 90a
set(CMAKE_CUDA_ARCHITECTURES 90a)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a")
include_directories("/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd/include")

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(benchmarking)
