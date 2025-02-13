#include "UtilLib/utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "MatrixLib/generations.cuh"

__global__ void generateHPCGProblem_kernel(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values,
    double * y    
) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_rows = nx * ny * nz;

    // each thread takes a row to fill
    // the row_ptr is already correctly filled
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        int nnz_i = 0;
        int current_index = row_ptr[i];

        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                col_idx[current_index] = j;
                                double current_val = i == j ? 26.0 : -1.0;
                                values[current_index] = current_val;
                                nnz_i++;
                                current_index++;
                            }
                        }
                    }
                }
            }
        }
        y[i] = 26.0 - nnz_i;
    }

}

__global__ void generateHPCGMatrix_kernel(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_rows = nx * ny * nz;

    // each thread takes a row to fill
    // the row_ptr is already correctly filled
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        int nnz_i = 0;
        int current_index = row_ptr[i];

        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                // if(threadIdx.x ==96 && blockIdx.x==8172){
                                //     printf("i: %d, ix: %d, iy: %d, iz: %d, sx: %d, sy: %d, sz: %d, current_index: %d\n", i, ix, iy, iz, sx, sy, sz, current_index);
                                // }
                                int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                col_idx[current_index] = j;
                                double current_val = i == j ? 26.0 : -1.0;
                                values[current_index] = current_val;
                                nnz_i++;
                                // if(current_index > row_ptr[i+1]){
                                //     printf("i: %d, ix: %d, iy: %d, iz: %d, sx: %d, sy: %d, sz: %d, current_index: %d\n", i, ix, iy, iz, sx, sy, sz, current_index);
                                // }
                                current_index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void get_num_elem_per_row_kernel(
    int nx, int ny, int nz,
    int * num_elem_per_row
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_rows = nx * ny * nz;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        int nnz_i = 0;
        // int row = ix + nx * iy + nx * ny * iz;

        // if (row != i){
        //     printf("row: %d, i: %d\n", row, i);
        // }

        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                nnz_i++;
                            }
                        }
                    }
                }
            }
        }
        // if(i == 48){
        //     printf("nnz_i: %d\n", nnz_i);
        // }
        num_elem_per_row[i + 1] = nnz_i;
    }
}

__global__ void iterative_sum_kernel(
    int * array, int num_elements
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0){
        for (int i = 1; i <= num_elements; i++){
            array[i] += array[i - 1];
        }
    }
}

void generateHPCGProblem(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values,
    double * y    
) {

    int num_rows = nx * ny * nz;

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    get_num_elem_per_row_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr + 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    iterative_sum_kernel<<<1, 1>>>(row_ptr, num_rows);
    CHECK_CUDA(cudaDeviceSynchronize());

    generateHPCGProblem_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr, col_idx, values, y);
    CHECK_CUDA(cudaDeviceSynchronize());
   
}

void generateHPCGMatrix(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values
){
    int num_rows = nx * ny * nz;

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    get_num_elem_per_row_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    iterative_sum_kernel<<<1, 1>>>(row_ptr, num_rows);
    CHECK_CUDA(cudaDeviceSynchronize());

    // print the row_ptr
    // grab row_ptr from device
    // int * row_ptr_host = (int *) malloc((num_rows + 1) * sizeof(int));
    // CHECK_CUDA(cudaMemcpy(row_ptr_host, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < num_rows + 1; i++){
    //     std::cout << row_ptr_host[i] << " ";
    // }
    // std::cout << std::endl;

    generateHPCGMatrix_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr, col_idx, values);
    CHECK_CUDA(cudaDeviceSynchronize());
}
