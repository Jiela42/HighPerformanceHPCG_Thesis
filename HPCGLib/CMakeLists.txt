# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name
project(HPCGlib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 90a)

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(benchmarking)
