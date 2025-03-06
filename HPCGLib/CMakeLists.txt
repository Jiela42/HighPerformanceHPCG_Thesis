# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(HPCGlib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(benchmarking)
