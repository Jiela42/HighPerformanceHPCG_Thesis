#include "HPCG_versions/cusparse.hpp"
#include "UtilLib/cuda_utils.hpp"
#include <iostream>


template <typename T>
void cuSparse_Implementation<T>::cusparse_computeSPMV(
    sparse_CSR_Matrix<T>& A,
    T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();
    int * A_row_ptr_d = A.get_row_ptr_d();
    int * A_col_idx_d = A.get_col_idx_d();
    T * A_values_d = A.get_values_d();

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


// Explicit template instantiation
template class cuSparse_Implementation<double>;
// template class cuSparse_Implementation<float>;