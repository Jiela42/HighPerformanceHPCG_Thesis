#include "HPCG_versions/banded_coloring.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include "MatrixLib/coloring.cuh"
// #include <iostream>
// #include <cuda_runtime.h>

__global__ void banded_coloring_SymGS_forward_kernel(){}


__global__ void banded_coloring_SymGS_backward_kernel(){}


__global__ void compute_num_colors_per_row(int num_rows, int max_num_colors, int * colors, int * num_colors_per_row){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = tid; i < max_num_colors; i += gridDim.x * blockDim.x){
        int num_rows_per_i = 0;
        for(int j = 0; j < num_rows; j++){
            if(colors[j] == i){
                num_rows_per_i++;
            }
        }
        num_colors_per_row[i] = num_rows_per_i;
    }

}
template <typename T>
void banded_coloring_Implementation<T>::banded_coloring_computeSymGS(
    banded_Matrix<T> & A, // we pass A for the metadata
    T * banded_A_d, // the data of matrix A is already on the device
    int num_rows, int num_cols,
    int num_bands, // the number of bands in the banded matrix
    int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
    T * x_d, T * y_d // the vectors x and y are already on the device
){
    int diag_offset = A.get_diag_index();
    assert(num_bands == A.get_num_bands());
    assert(num_rows == A.get_num_rows());
    assert(num_cols == A.get_num_cols());
    assert(diag_offset >= 0);

    // first we need to color the matrix
    // we make a device vector for the colors
    int * colors_d;

    // we allocate the memory for the colors
    CHECK_CUDA(cudaMalloc(&colors_d, num_rows * sizeof(int)));

    // we initialize the colors to -1
    CHECK_CUDA(cudaMemset(colors_d, -1, num_rows * sizeof(int)));

    color_for_forward_pass_kernel<<<1, 1024>>>(num_rows, num_bands, diag_offset, banded_A_d, j_min_i, colors_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    int max_color;
    CHECK_CUDA(cudaMemcpy(&max_color, &colors_d[num_rows-1], sizeof(int), cudaMemcpyDeviceToHost));

    for(int color = 0; color < max_color; color++){
        // we need to do a forward pass
        banded_coloring_SymGS_forward_kernel<<<1, 1024>>>();
        CHECK_CUDA(cudaDeviceSynchronize());
    }


}

// explicit template instantiation
template class banded_coloring_Implementation<double>;