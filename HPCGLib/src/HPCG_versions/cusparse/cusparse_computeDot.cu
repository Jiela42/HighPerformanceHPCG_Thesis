#include "HPCG_versions/cusparse.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <cublas_v2.h>

__global__ void elem_wise_mult_of_vectors_kernel(int num_rows, double *x, double *y, double *z){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x){
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
    cublasHandle_t handle;
    cublasCreate(&handle);

    int incx = 1;
    int incy = 1;
    double result_host = 0.0;

    CHECK_CUBLAS(cublasDdot(handle, num_rows, x_d, incx, y_d, incy, &result_host));

    CHECK_CUDA(cudaDeviceSynchronize());

    // printf("result_host: %f\n", result_host);

    CHECK_CUDA(cudaMemcpy(z_d, &result_host, sizeof(T), cudaMemcpyHostToDevice));

    cublasDestroy(handle);


}

template class cuSparse_Implementation<double>;