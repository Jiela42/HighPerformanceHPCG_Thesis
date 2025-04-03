#include "HPCG_versions/cusparse_hipified.hpp"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/utils_hipified.cuh"
#include "hip/hip_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"
#include <iostream>
#include "hipblas/hipblas.h"


__global__ void elem_wise_mult_of_vectors_kernel(local_int_t num_rows, DataType *x, DataType *y, DataType *z){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x){
        z[row] = x[row] * y[row];
    }
}

template <typename T>
void cuSparse_Implementation<T>::cusparse_computeDot(
    sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
    T * x_d,
    T * y_d,
    T * z_d
    ){
    // we compute z = xy

    int num_rows = A.get_num_rows();

    // use cuBLAS to compute the dot product
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    int incx = 1;
    int incy = 1;
    DataType result_host = 0.0;

    CHECK_CUBLAS(hipblasDdot(handle, num_rows, x_d, incx, y_d, incy, &result_host));

    CHECK_CUDA(hipDeviceSynchronize());

    // printf("result_host: %f\n", result_host);

    CHECK_CUDA(hipMemcpy(z_d, &result_host, sizeof(T), hipMemcpyHostToDevice));

    hipblasDestroy(handle);


}

template class cuSparse_Implementation<DataType>;