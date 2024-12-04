#include "HPCG_versions/naiveBanded.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>


__global__ void naiveBanded_SPMV_kernel(
        double* banded_A,
        int num_rows, int num_bands, int * j_min_i,
        double* x, double* y
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // every thread computes one or more rows of the matrix
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int band = 0; band < num_bands; band++) {
            int j = i + j_min_i[band];
            int current_row = i * num_bands;
            if (j >= 0 && j < num_rows) {
                sum_i += banded_A[current_row + band] * x[j];
            }
        }
        y[i] = sum_i;
    }
}
    

template <typename T>
void naiveBanded_Implementation<T>::naiveBanded_computeSPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {
        // call the kernel for the naive banded SPMV
        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = NUM_CORES_PER_SM * 4;
        int num_blocks = std::min(NUM_PHYSICAL_CORES, ceiling_division(num_rows, num_threads));

        assert(num_bands == A.get_num_bands());
        assert(num_rows == A.get_num_rows());
        assert(num_cols == A.get_num_cols());

        // call the kernel
        naiveBanded_SPMV_kernel<<<num_blocks, num_threads>>>(
            banded_A_d, num_rows, num_bands, j_min_i, x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();
    }

// explicit template instantiation
template class naiveBanded_Implementation<double>;