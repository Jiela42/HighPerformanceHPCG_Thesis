#include "HPCG_versions/cusparse_hipified.hpp"
#include "UtilLib/cuda_utils_hipified.hpp"
#include <iostream>
//#include <hipsparse/hipsarse.h>

template <typename T>
void cuSparse_Implementation<T>::cusparse_computeSPMV(
    sparse_CSR_Matrix<T>& A,
    T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    global_int_t nnz = A.get_nnz();
    local_int_t * A_row_ptr_d = A.get_row_ptr_d();
    local_int_t * A_col_idx_d = A.get_col_idx_d();
    T * A_values_d = A.get_values_d();

    // Determine the CUDA data type based on T
    hipDataType cuda_dtype;
    if constexpr (std::is_same<T, double>::value) {
        cuda_dtype = HIP_R_64F; // Double precision
    } else if constexpr (std::is_same<T, float>::value) {
        cuda_dtype = HIP_R_32F; // Single precision
    } else {
        static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value, "Unsupported data type");
    }

    hipsparseIndexType_t cuda_idx_dtype;
    if constexpr(std::is_same<local_int_t, int>::value) {
        cuda_idx_dtype = HIPSPARSE_INDEX_32I; // 32-bit integer
    } else if constexpr(std::is_same<local_int_t, long>::value) {
        cuda_idx_dtype = HIPSPARSE_INDEX_64I; // 64-bit integer
    } else {
        static_assert(std::is_same<local_int_t, int>::value || std::is_same<local_int_t, long>::value, "Unsupported index type");
    }

    hipsparseHandle_t handle;
    hipsparseCreate(&handle);

    T alpha = 1.0;
    T beta = 0.0;


    // Create the matrix and vector descriptors
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(hipsparseCreateCsr(&matA, num_rows, num_cols, nnz, A_row_ptr_d, A_col_idx_d, A_values_d,
                                    cuda_idx_dtype, cuda_idx_dtype, HIPSPARSE_INDEX_BASE_ZERO, cuda_dtype));
    CHECK_CUSPARSE(hipsparseCreateDnVec(&vecX, num_cols, x_d, cuda_dtype));
    CHECK_CUSPARSE(hipsparseCreateDnVec(&vecY, num_rows, y_d, cuda_dtype));

    // Allocate buffer
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, cuda_dtype, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(hipMalloc(&dBuffer, bufferSize));

    // Perform SpMV
    CHECK_CUSPARSE(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, cuda_dtype, HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer));


    hipsparseDestroy(handle);
}


// Explicit template instantiation
template class cuSparse_Implementation<DataType>;
// template class cuSparse_Implementation<float>;