#include "HPCG_versions/naiveBanded.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"

#include "cuda_utils.hpp"

template <typename T>
void naiveBanded_Implementation<T>::naiveBanded_compute_SPMV(
        const sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {
        // call the kernel for the naive banded SPMV
        // since every thread is working on one or more rows we need to base the number of threads on that


    }

// explicit template instantiation
template class naiveBanded_Implementation<double>;