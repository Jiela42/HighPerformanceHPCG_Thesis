#include "HPCG_versions/striped_warp_reduction.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>


__global__ void striped_warp_reduction_SPMV_kernel(
        double* striped_A,
        int num_rows, int num_stripes, int * j_min_i,
        double* x, double* y
    )
{
    // printf("striped_warp_reduction_SPMV_kernel\n");
    int cooperation_number = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % cooperation_number;

    // every thread computes one or more rows of the matrix
    for (int i = tid/cooperation_number; i < num_rows; i += (blockDim.x * gridDim.x)/cooperation_number) {
        // compute the matrix-vector product for the ith row
        double sum_i = 0;
        for (int stripe = lane; stripe < num_stripes; stripe += cooperation_number) {
            int j = i + j_min_i[stripe];
            int current_row = i * num_stripes;
            if (j >= 0 && j < num_rows) {
                sum_i += striped_A[current_row + stripe] * x[j];
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
void striped_warp_reduction_Implementation<T>::striped_warp_reduction_computeSPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

        // std::cout << "striped_warp_reduction_computeSPMV" << std::endl;

        int num_rows = A.get_num_rows();
        int num_stripes = A.get_num_stripes();
        int * j_min_i = A.get_j_min_i_d();
        T * striped_A_d = A.get_values_d();

        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = 1024;
        int rows_per_block = num_threads / 4;
        int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, rows_per_block));

        // call the kernel
        striped_warp_reduction_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, num_rows, num_stripes, j_min_i, x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();

        // std::cerr << "Assertion failed in function: " << __PRETTY_FUNCTION__ << std::endl;
        // assert(false);
    }

// explicit template instantiation
template class striped_warp_reduction_Implementation<double>;