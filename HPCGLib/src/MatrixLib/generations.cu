#include "UtilLib/utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "MatrixLib/generations.cuh"
#include "UtilLib/hpcg_mpi_utils.cuh"

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

__global__ void generate_f2c_operator_kernel(
    int nxf, int nyf, int nzf,
    int nxc, int nyc, int nzc,
    int * f2c_op
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // int num_fine_rows = nxf * nyf * nzf;
    int num_coarse_rows = nxc * nyc * nzc;

    for(int coarse_idx = tid; coarse_idx < num_coarse_rows; coarse_idx += blockDim.x * gridDim.x){
        int izc = coarse_idx / (nxc * nyc);
        int iyc = (coarse_idx % (nxc * nyc)) / nxc;
        int ixc = coarse_idx % nxc;

        int izf = izc * 2;
        int iyf = iyc * 2;
        int ixf = ixc * 2;

        int fine_idx = ixf + nxf * iyf + nxf * nyf * izf;
        f2c_op[coarse_idx] = fine_idx;

        // if(coarse_idx < 5){
        //     printf("coarse_idx: %d, fine_idx: %d\n", coarse_idx, fine_idx);
        // }
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


void generate_f2c_operator(
    int nxc, int nyc, int nzc,
    int nxf, int nyf, int nzf,
    int * f2c_op
){

    int num_coarse_rows = nxc * nyc * nzc;

    int num_threads = 1024;
    int num_blocks = num_coarse_rows/num_threads;

    // we need at least 1 block
    num_blocks = num_blocks == 0 ? 1 : num_blocks;

    generate_f2c_operator_kernel<<<num_blocks, num_threads>>>(nxf, nyf, nzf, nxc, nyc, nzc, f2c_op);
    CHECK_CUDA(cudaDeviceSynchronize());
}


__global__ void GenerateStripedPartialMatrix_kernel(int nx, int ny, int nz, int gnx, int gny, int gnz, int offset_x, int offset_y, int offset_z, double *A){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    local_int_t num_rows = nx * ny * nz;
    local_int_t num_cols = nx * ny * nz;
    
    for (int i=tid; i<num_rows; i += blockDim.x * gridDim.x) {
        int gx = i % nx + offset_x;
        int gy = (i / nx) % ny + offset_y;
        int gz = i / (nx * ny) + offset_z;

        for (int sz = -1; sz < 2; sz++){
            for(int sy = -1; sy < 2; sy++){
                for(int sx = -1; sx < 2; sx++){

                    if(gx + sx < 0 || gx + sx >= gnx ||
                        gy + sy < 0 || gy + sy >= gny ||
                        gz + sz < 0 || gz + sz >= gnz) {
                            *A = 0.0;
                            A++;
                        } else {
                            if(sx == 0 && sy == 0 && sz == 0){
                                *A = 26.0;
                                A++;
                            } else {
                                *A = -1.0;
                                A++;
                            }
                        }
                }
            }
        }



    }
}


__global__ void generate_partialf2c_operator_kernel(
    int nxf, int nyf, int nzf,
    int nxc, int nyc, int nzc,
    int * f2c_op_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // int num_fine_rows = nxf * nyf * nzf;
    int num_coarse_rows = nxc * nyc * nzc;

    for(int coarse_idx = tid; coarse_idx < num_coarse_rows; coarse_idx += blockDim.x * gridDim.x){
        int izc = coarse_idx / (nxc * nyc);
        int iyc = (coarse_idx % (nxc * nyc)) / nxc;
        int ixc = coarse_idx % nxc;

        int izf = izc * 2;
        int iyf = iyc * 2;
        int ixf = ixc * 2;

        int fine_idx = ixf + nxf * iyf + nxf * nyf * izf;
        f2c_op_d[coarse_idx] = fine_idx;

        // if(coarse_idx < 5){
        //     printf("coarse_idx: %d, fine_idx: %d\n", coarse_idx, fine_idx);
        // }
    }
}


void GenerateStripedPartialMatrix_GPU(Problem *problem, double *A_d) {
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    local_int_t num_rows = nx * ny * nz;

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    GenerateStripedPartialMatrix_kernel<<<block_size, num_blocks>>>(problem->nx, problem->ny, problem->nz, problem->gnx, problem->gny, problem->gnz, problem->gx0, problem->gy0, problem->gz0, A_d);
}

void generate_partialf2c_operator(
    int nxc, int nyc, int nzc,
    int nxf, int nyf, int nzf,
    int * f2c_op_d
){

    int num_coarse_rows = nxc * nyc * nzc;

    int num_threads = 1024;
    int num_blocks = num_coarse_rows/num_threads;

    // we need at least 1 block
    num_blocks = num_blocks == 0 ? 1 : num_blocks;

    generate_partialf2c_operator_kernel<<<num_blocks, num_threads>>>(nxf, nyf, nzf, nxc, nyc, nzc, f2c_op_d);
    CHECK_CUDA(cudaDeviceSynchronize());
}


