#include "HPCG_versions/naiveBanded.cuh"

// #include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <cmath>
// #include "utils.cuh"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>
// #include <cuda.h>


int ceiling_division(int numerator, int denominator) {
    return static_cast<int>(std::ceil(static_cast<double>(numerator) / denominator));
}


template <typename T>
void naiveBanded_Implementation<T>::naiveBanded_computeSPMV(
        const sparse_CSR_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {
        // call the kernel for the naive banded SPMV
        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = MAX_THREADS_PER_BLOCK;
        int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, num_threads));


        // call the kernel
        naiveBanded_SPMV_kernel<<<num_blocks, num_threads>>>(
            banded_A_d, num_rows, num_bands, j_min_i, x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();
    }

// explicit template instantiation
template class naiveBanded_Implementation<double>;