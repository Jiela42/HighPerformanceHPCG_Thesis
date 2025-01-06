# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(HPCGlib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)

# Also include the AMGX library
include_directories(../../AMGX/include)

# Find the AMGX library
find_library(AMGX_LIB amgx PATHS ../../AMGX/build NO_DEFAULT_PATH)
if(NOT AMGX_LIB)
    message(FATAL_ERROR "AMGX library not found. Please check the path.")
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(benchmarking)
