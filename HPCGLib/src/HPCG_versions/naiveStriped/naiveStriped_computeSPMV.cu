#include "HPCG_versions/naiveStriped.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>


__global__ void naiveStriped_SPMV_kernel(
        DataType* striped_A,
        local_int_t num_rows, int num_stripes, local_int_t * j_min_i,
        DataType* x, DataType* y
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // every thread computes one or more rows of the matrix
    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int stripe = 0; stripe < num_stripes; stripe++) {
            local_int_t j = i + j_min_i[stripe];
            local_int_t current_row = i * num_stripes;
            if (j >= 0 && j < num_rows) {
                sum_i += striped_A[current_row + stripe] * x[j];
            }
        }
        y[i] = sum_i;
    }
}
    

template <typename T>
void naiveStriped_Implementation<T>::naiveStriped_computeSPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

        local_int_t num_rows = A.get_num_rows();
        int num_stripes = A.get_num_stripes();
        local_int_t * j_min_i = A.get_j_min_i_d();
        T * striped_A_d = A.get_values_d();

        // call the kernel for the naive striped SPMV
        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = NUM_CORES_PER_SM * 4;
        int num_blocks = std::min(NUM_PHYSICAL_CORES, ceiling_division(num_rows, num_threads));

        // call the kernel
        naiveStriped_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, num_rows, num_stripes, j_min_i, x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();
    }

// explicit template instantiation
template class naiveStriped_Implementation<double>;