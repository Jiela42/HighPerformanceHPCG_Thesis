#include "HPCG_versions/no_store_striped_coloring.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include "MatrixLib/coloring.cuh"
// #include <iostream>
// #include <cuda_runtime.h>

__global__ void no_store_striped_coloring_half_SymGS_kernel(
    int color,
    int nx, int ny, int nz,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    double * x, double * y
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for(int xy = 0 + warp_id; xy < nx*ny; xy += num_warps){
        int xi = xy / ny;
        int yi = xy % ny;

        int enumerator = color - xi - 2*yi;

        if(enumerator < 0){
            continue;
        }
        if(enumerator % 4 != 0){
            continue;
        }
        int zi = enumerator / 4;

        if (zi < nz){
            int row = xi + yi * nx + zi * nx * ny;
            double my_sum = 0.0;
            for(int stripe = lane_id; stripe < num_stripes; stripe += WARP_SIZE){

                int col = j_min_i[stripe] + row;
                double val = striped_A[row * num_stripes + stripe];
                if(col < num_cols && col >= 0){
                    my_sum -= val * x[col];
                }
            }

            // reduce the my_sum using warp reduction
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
                my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
            }

            __syncthreads();
            if (lane_id == 0){
                double diag = striped_A[row * num_stripes + diag_offset];
                double sum = diag * x[row] + y[row] + my_sum;          
                x[row] = sum / diag;
            }
            __syncthreads();

        }
    }
}


template <typename T>
void no_store_striped_coloring_Implementation<T>::no_store_striped_coloring_computeSymGS(
    striped_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device
){

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    int * j_min_i = A.get_j_min_i_d();
    double * striped_A_d = A.get_values_d();

    int diag_offset = A.get_diag_index();

    assert(diag_offset >= 0);
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();

    // std::cout << "nx: " << nx << ", ny: " << ny << ", nz: " << nz << std::endl;

    // double looki_1082 = 0.0;
    // CHECK_CUDA(cudaMemcpy(&looki_1082, &x_d[1082], sizeof(double), cudaMemcpyDeviceToHost));

    // std::cout << "beginning of symgs: " << looki_1082 << std::endl;

    // the number of blocks is now dependent on the maximum number of rows per color

    int max_num_rows_per_color = std::min(nx * ny / 4, std::min(nx * nz / 2, ny * nz));
    int max_color = (nx-1) + 2 * (ny-1) + 4 * (nz-1);

    int num_blocks = std::min(ceiling_division(max_num_rows_per_color, 1024/WARP_SIZE), MAX_NUM_BLOCKS);
    for(int color = 0; color <= max_color; color++){
        // we need to do a forward pass
        // std::cout << "color: " << color << std::endl;
        no_store_striped_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
        color,
        nx, ny, nz,
        num_rows, num_cols,
        num_stripes, diag_offset,
        j_min_i,
        striped_A_d,
        x_d, y_d
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // we need to do a backward pass,
    // the colors for this are the same just in reverse order
    
    for(int color = max_color; color  >= 0; color--){

        no_store_striped_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
        color,
        nx, ny, nz,
        num_rows, num_cols,
        num_stripes, diag_offset,
        j_min_i,
        striped_A_d,
        x_d, y_d
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // CHECK_CUDA(cudaMemcpy(&looki_1082, &x_d[1082], sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "end of symgs: " << looki_1082 << std::endl;
    
}

// explicit template instantiation
template class no_store_striped_coloring_Implementation<double>;