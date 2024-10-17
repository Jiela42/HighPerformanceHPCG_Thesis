#include "HPCG_versions/cusparse.hpp"
#include "cuda_utils.hpp"
#include <iostream>

// template <typename T>
// void cuSparse_Implementation<T>::compute_SPMV(const sparse_CSR_Matrix<T>& A, const std::vector<T>& x, std::vector<T>& y) {
//     cusparse_computeSPMV(A, x, y);
// }

template <typename T>
void cuSparse_Implementation<T>::cusparse_computeSPMV(const sparse_CSR_Matrix<T>& A, const std::vector<T>& x, std::vector<T>& y) {
    int * A_row_ptr_d;
    int * A_col_idx_d;
    T * A_values_d;
    T * x_d;
    T * y_d;

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A.row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A.col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A.values.data(), nnz * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(T), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    T alpha = 1.0;
    T beta = 0.0;

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    if constexpr (std::is_same<T, double>::value) {
        cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_cols, nnz, &alpha, descr, A_values_d, A_row_ptr_d, A_col_idx_d, x_d, &beta, y_d);
    } else if constexpr (std::is_same<T, float>::value) {
        cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_cols, nnz, &alpha, descr, A_values_d, A_row_ptr_d, A_col_idx_d, x_d, &beta, y_d);
    }

    CHECK_CUDA(cudaMemcpy(y.data(), y_d, num_rows * sizeof(T), cudaMemcpyDeviceToHost));

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
}

// Explicit template instantiation
template class cuSparse_Implementation<double>;
// template class cuSparse_Implementation<float>;