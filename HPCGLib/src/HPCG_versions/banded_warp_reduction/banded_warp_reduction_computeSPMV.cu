#include "HPCG_versions/banded_warp_reduction.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>


__global__ void banded_warp_reduction_SPMV_kernel(
        double* banded_A,
        int num_rows, int num_bands, int * j_min_i,
        double* x, double* y
    )
{
    
    int cooperation_number = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % cooperation_number;

    // every thread computes one or more rows of the matrix
    for (int i = tid/cooperation_number; i < num_rows; i += (blockDim.x * gridDim.x)/cooperation_number) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int band = lane; band < num_bands; band += cooperation_number) {
            int j = i + j_min_i[band];
            int current_row = i * num_bands;
            if (j >= 0 && j < num_rows) {
                sum_i += banded_A[current_row + band] * x[j];
            }
        }

        // now let's reduce the sum_i to a single value using warp-level reduction
        for(int offset = cooperation_number/2; offset > 0; offset /= 2){
            sum_i += __shfl_down_sync(0xFFFFFFFF, sum_i, offset);
        }

        __syncthreads();

        if (lane == 0){
            y[i] = sum_i;
        }
    }
}

template <typename T>
void banded_warp_reduction_Implementation<T>::banded_warp_reduction_computeSPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {
        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = 1024;
        int rows_per_block = num_threads / 4;
        int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, rows_per_block));

        assert(num_bands == A.get_num_bands());
        assert(num_rows == A.get_num_rows());
        assert(num_cols == A.get_num_cols());

        // call the kernel
        banded_warp_reduction_SPMV_kernel<<<num_blocks, num_threads>>>(
            banded_A_d, num_rows, num_bands, j_min_i, x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();

        // std::cerr << "Assertion failed in function: " << __PRETTY_FUNCTION__ << std::endl;
        // assert(false);
    }

// explicit template instantiation
template class banded_warp_reduction_Implementation<double>;