#include "HPCG_versions/cusparse.hpp"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>
#include <cublas_v2.h>


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

    int incx, incy = 1;
    double result_host = 0.0;

    CHECK_CUBLAS(cublasDdot(handle, num_rows, x_d, incx, y_d, incy, &result_host));

    CHECK_CUDA(cudaDeviceSynchronize());

    printf("result_host: %f\n", result_host);

    CHECK_CUDA(cudaMemcpy(z_d, &result_host, sizeof(T), cudaMemcpyHostToDevice));

    cublasDestroy(handle);

}

template class cuSparse_Implementation<double>;