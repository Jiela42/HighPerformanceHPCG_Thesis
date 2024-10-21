#include "HPCG_versions/cusparse.hpp"
#include "cuda_utils.hpp"
#include <iostream>


template <typename T>
void cuSparse_Implementation<T>::cusparse_computeSPMV(
    const sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
    T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    T alpha = 1.0;
    T beta = 0.0;


    // Create the matrix and vector descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, num_rows, num_cols, nnz, A_row_ptr_d, A_col_idx_d, A_values_d,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, x_d, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, y_d, CUDA_R_64F));

    // Allocate buffer
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));


    cusparseDestroy(handle);
}

template <typename T>
cuSparse_Implementation<T>cusparse_computeSPMV_with_datacopy(const sparse_CSR_Matrix<T>& A, const std::vector<T>& x, std::vector<T>& y) {
    int * A_row_ptr_d;
    int * A_col_idx_d;
    T * A_values_d;
    T * x_d;
    T * y_d;

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    const int * A_row_ptr_data = A.get_row_ptr().data();
    const int * A_col_idx_data = A.get_col_idx().data();
    const double * A_values_data = A.get_values().data();

    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(T), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    T alpha = 1.0;
    T beta = 0.0;


    // Create the matrix and vector descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, num_rows, num_cols, nnz, A_row_ptr_d, A_col_idx_d, A_values_d,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, x_d, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, y_d, CUDA_R_64F));

    // Allocate buffer
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));


    CHECK_CUDA(cudaMemcpy(y.data(), y_d, num_rows * sizeof(T), cudaMemcpyDeviceToHost));

    cusparseDestroy(handle);

    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
}

// Explicit template instantiation
template class cuSparse_Implementation<double>;
// template class cuSparse_Implementation<float>;