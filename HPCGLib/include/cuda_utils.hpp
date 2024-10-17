#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <iostream>

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