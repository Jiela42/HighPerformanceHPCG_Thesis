#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipsparse/hipsparse.h>
#include <iostream>

// here we define some machiene dependent constants
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_NUM_BLOCKS 65535
#define NUM_PHYSICAL_CORES 10496
#define NUM_SM 84
#define NUM_CORES_PER_SM 128
#define WARP_SIZE 32
// this is KB per SM
#define SHARED_MEM_SIZE 100

// Macro to check CUDA function calls
#define CHECK_CUDA(call) \
    { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                      << hipGetErrorString(err) << "." << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Macro to check cuSPARSE function calls
#define CHECK_CUSPARSE(call) \
    { \
        hipsparseStatus_t err = call; \
        if (err != HIPSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                      << hipsparseGetErrorString(err) << "." << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Macro to check cuBLAS function calls
#define CHECK_CUBLAS(call) \
    { \
        hipblasStatus_t err = call; \
        if (err != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                      << err << "." << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#endif // CUDA_UTILS_HPP