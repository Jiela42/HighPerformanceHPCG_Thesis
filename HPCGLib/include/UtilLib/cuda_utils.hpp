#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <iostream>

// here we define some machiene dependent constants
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_NUM_BLOCKS 65535
#define NUM_PHYSICAL_CORES 10496
#define NUM_SM 84
#define NUM_CORES_PER_SM 128

// Macro to check CUDA function calls
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << "." << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Macro to check cuSPARSE function calls
#define CHECK_CUSPARSE(call) \
    { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                      << cusparseGetErrorString(err) << "." << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#endif // CUDA_UTILS_HPP