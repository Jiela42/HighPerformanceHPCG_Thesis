#include "UtilLib/utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "MatrixLib/generations.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

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

__global__ void row_ptr_from_striped(
    int num_rows,
    int num_stripes,
    int * j_min_i,
    double * values,
    int * num_nnz_i
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        int nnz_i = 0;
        for(int stripe_j = 0; stripe_j < num_stripes; stripe_j++){
            int j = j_min_i[stripe_j] + i;
            // check if j is in bounds (since not every point has all 27 neighbours)
            if( j >= 0 && j < num_rows && values[i* num_stripes + stripe_j] != 0.0){
                nnz_i++;
            }
        }
        num_nnz_i[i+1] = nnz_i;
    }
}

__global__ void col_and_val_from_striped(
    // row_ptr is already set
    int num_rows,
    int num_stripes,
    int * j_min_i,
    double * striped_A_d,
    int * row_ptr, int * col_idx, double * values
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        int current_index = row_ptr[i];
        for(int stripe_j = 0; stripe_j < num_stripes; stripe_j++){
            int j = j_min_i[stripe_j] + i;
            double val = striped_A_d[i* num_stripes + stripe_j];
            // check if j is in bounds (since not every point has all 27 neighbours)
            if( j >= 0 && j < num_rows && val != 0.0){
                col_idx[current_index] = j;
                values[current_index] = val;
                current_index++;
            }
        }
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

__global__ void generate_striped_from_CSR_kernel(
    int num_rows,
    int num_stripes,
    int * j_min_i,
    int * row_ptr, int * col_idx, double * values,
    double * striped_A_d,
    int * num_nnz_i
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        int nnz_i = 0;
        for(int stripe_j = 0; stripe_j < num_stripes; stripe_j++){
            int j = j_min_i[stripe_j] + i;
            // check if j is in bounds (since not every point has all 27 neighbours)
            if( j >= 0 && j < num_rows){
                double elem = 0.0;
                // we gotta find the element in the CSR matrix
                for(int r = row_ptr[i]; r < row_ptr[i+1]; r++){
                    if(col_idx[r] == j){
                        elem = values[r];
                        break;
                    }
                }
                if(elem != 0.0){
                    striped_A_d[i* num_stripes + stripe_j] = elem;
                    // printf("i: %d, j: %d, elem: %f, stripe_j: %d \n", i, j, elem, stripe_j);
                    nnz_i++;
                }
            }
        }
        num_nnz_i[i] = nnz_i;
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

// we return how many non-zeros we ended up copying
int generate_striped_3D27P_Matrix_from_CSR(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values,
    int num_stripes, int * j_min_i,
    double * striped_A_d
){

    // allocate space for non-zeros
    int num_rows = nx * ny * nz;
    int * num_nnz_i;
    CHECK_CUDA(cudaMalloc(&num_nnz_i, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(num_nnz_i, 0, num_rows * sizeof(int)));

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    generate_striped_from_CSR_kernel<<<num_blocks, block_size>>>(num_rows, num_stripes, j_min_i, row_ptr, col_idx, values, striped_A_d, num_nnz_i);

    CHECK_CUDA(cudaDeviceSynchronize());
    
    // reduce using thrust
    thrust::device_ptr<int> num_nnz_i_ptr(num_nnz_i);
    int total_nnz = thrust::reduce(num_nnz_i_ptr, num_nnz_i_ptr + num_rows);

    // free memory
    CHECK_CUDA(cudaFree(num_nnz_i));

    return total_nnz;

}

// we return how many non-zeros we ended up copying
int generate_CSR_from_Striped(
    int num_rows, int num_stripes,
    int * j_min_i, double * striped_A_d,
    int * row_ptr, int * col_idx, double * values
){
    // first we set the row_ptr
    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    // set the row_ptr first
    row_ptr_from_striped<<<num_blocks, block_size>>>(num_rows, num_stripes, j_min_i, striped_A_d, row_ptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // we need to do the prefix sum
    iterative_sum_kernel<<<1, 1>>>(row_ptr, num_rows);
    CHECK_CUDA(cudaDeviceSynchronize());

    // read the last element of row_ptr
    int num_nnz = 0;
    CHECK_CUDA(cudaMemcpy(&num_nnz, row_ptr + num_rows, sizeof(int), cudaMemcpyDeviceToHost));

    // now we generate the col_idx and values
    col_and_val_from_striped<<<num_blocks, block_size>>>(num_rows, num_stripes, j_min_i, striped_A_d, row_ptr, col_idx, values);
    CHECK_CUDA(cudaDeviceSynchronize());

    return num_nnz;

}









