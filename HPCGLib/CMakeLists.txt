# Minimum CMake version required
cmake_minimum_required(VERSION 3.30)

# Project name
project(HPCGlib)
enable_language(HIP)

#set(CMAKE_HIP_ARCHITECTURES "gfx942")
# Set CUDA architecture to 90a
# set(CMAKE_CUDA_ARCHITECTURES 90a 86)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
#link_directories("/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd/lib")
add_compile_definitions(__HIP_PLATFORM_AMD__)
add_compile_definitions(__HIP_PLATFORM_HCC__)
set(HIP_ARCHITECTURES "gfx942")
set(CMAKE_HIP_ARCHITECTURES "gfx942")
add_compile_options("--offload-arch=gfx942")
add_compile_definitions(HIP_ENABLE_WARP_SYNC_BUILTINS)

find_package(HIP REQUIRED)
find_package(hipsparse REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocthrust REQUIRED)
find_package(rocprim REQUIRED)
find_package(rccl REQUIRED)
include_directories(${HIP_INCLUDE_DIRS}          # Include directories for HIP
${hipSPARSE_INCLUDE_DIRS}    # Include directories for HIPSPARSE
${hipBLAS_INCLUDE_DIRS}
${rocthrust_INCLUDE_DIRS}
${rocprim}
${rccl_INCLUDE_DIRS})
# Include HIP directories
#include_directories(${HIP_INCLUDE_DIRS} ${hipSPARSE_INCLUDE_DIRS})
#include_directories(${HIP_INCLUDE_DIRS} ${hipSPARSE_INCLUDE_DIRS} ${hipBLAS_INCLUDE_DIRS})

file(GLOB SOURCES "*.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

include_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hip-6.2.1-dnxqftoeqt7lshn7es5sf7vnjy2n65ep/include")
include_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipsparse-6.2.1-r2wk6eagqbyozhokoioobayvtjtgoilb/include")
include_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipblas-6.2.1-hxzi7r33xc4fmvxzz64rhdtbq44bhteu/include")
include_directories(${ROCTHRUST_INCLUDE_DIRS})
include_directories(${ROCPRIM_INCLUDE_DIRS})
link_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hip-6.2.1-dnxqftoeqt7lshn7es5sf7vnjy2n65ep/lib")
link_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipsparse-6.2.1-r2wk6eagqbyozhokoioobayvtjtgoilb/lib")
link_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipblas-6.2.1-hxzi7r33xc4fmvxzz64rhdtbq44bhteu/lib")
link_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/rocprim-6.2.1-lguqn2d2gznl7t5n7h5w3ndneigfr2zg/lib")
link_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/rocthrust-6.2.1-ihyylrpt7zc2buckjcgkaxlmll3kib4u/lib")
#link_libraries("-L/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hip-6.2.1-dnxqftoeqt7lshn7es5sf7vnjy2n65ep/lib/libamdhip64.so")
#link_libraries("amdhip64")
link_libraries(${ROCPRIM_LIBRARIES} hipsparse hipblas)
link_directories("/capstor/scratch/cscs/ybudanaz/jingjing/HighPerformanceHPCG_Thesis/HPCGLib/build> ls /capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/rccl-6.2.1-6ig3jji6qqna4oqyxohqmazhxts7i7wd/include/")
#find_library(AMDHIP64_LIBRARY
#    NAMES amdhip64
#    HINTS /capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hip-6.2.1-dnxqftoeqt7lshn7es5sf7vnjy2n65ep/lib
#)

#if (AMDHIP64_LIBRARY)
#    message(STATUS "Found AMDHIP64: ${AMDHIP64_LIBRARY}")
#    link_libraries(${AMDHIP64_LIBRARY})
#else()
#    message(FATAL_ERROR "AMDHIP64 not found!")
#endif()

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx942")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a")
#include_directories("/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd/include")
set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE HIP)


add_compile_options(-Wno-c++11-narrowing)

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(benchmarking)
