#include "hip/hip_runtime.h"
#include "UtilLib/utils_hipified.cuh"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "MatrixLib/generations_hipified.cuh"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"

#include <cmath>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

__global__ void generateHPCGProblem_kernel(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values,
    DataType * y    
) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);

    // each thread takes a row to fill
    // the row_ptr is already correctly filled
    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        local_int_t nnz_i = 0;
        local_int_t current_index = row_ptr[i];

        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                local_int_t j = static_cast<local_int_t>(ix + sx) + static_cast<local_int_t>(nx) * (iy + sy) + static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * (iz + sz);
                                col_idx[current_index] = j;
                                DataType current_val = i == j ? 26.0 : -1.0;
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
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);

    // each thread takes a row to fill
    // the row_ptr is already correctly filled
    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        local_int_t nnz_i = 0;
        local_int_t current_index = row_ptr[i];

        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                // if(threadIdx.x ==96 && blockIdx.x==8172){
                                //     printf("i: %d, ix: %d, iy: %d, iz: %d, sx: %d, sy: %d, sz: %d, current_index: %d\n", i, ix, iy, iz, sx, sy, sz, current_index);
                                // }
                                local_int_t j = static_cast<local_int_t>(ix + sx) + static_cast<local_int_t>(nx) * (iy + sy) + static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * (iz + sz);
                                col_idx[current_index] = j;
                                DataType current_val = i == j ? 26.0 : -1.0;
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
    local_int_t * num_elem_per_row
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        local_int_t nnz_i = 0;
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
    local_int_t num_rows,
    int num_stripes,
    local_int_t * j_min_i,
    DataType * values,
    local_int_t * num_nnz_i
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        local_int_t nnz_i = 0;
        for(int stripe_j = 0; stripe_j < num_stripes; stripe_j++){
            local_int_t j = j_min_i[stripe_j] + i;
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
    local_int_t num_rows,
    int num_stripes,
    local_int_t * j_min_i,
    DataType * striped_A_d,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        local_int_t current_index = row_ptr[i];
        for(int stripe_j = 0; stripe_j < num_stripes; stripe_j++){
            local_int_t j = j_min_i[stripe_j] + i;
            DataType val = striped_A_d[i* num_stripes + stripe_j];
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
    local_int_t * array, local_int_t num_elements
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0){
        for (int i = 1; i <= num_elements; i++){
            array[i] += array[i - 1];
        }
    }
}

__global__ void generate_striped_from_CSR_kernel(
    local_int_t num_rows,
    int num_stripes,
    local_int_t * j_min_i,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values,
    DataType * striped_A_d,
    local_int_t * num_nnz_i
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (local_int_t i = tid; i < num_rows; i += blockDim.x * gridDim.x){
        local_int_t nnz_i = 0;
        for(int stripe_j = 0; stripe_j < num_stripes; stripe_j++){
            local_int_t j = j_min_i[stripe_j] + i;
            // check if j is in bounds (since not every point has all 27 neighbours)
            if( j >= 0 && j < num_rows){
                DataType elem = 0.0;
                // we gotta find the element in the CSR matrix
                for(local_int_t r = row_ptr[i]; r < row_ptr[i+1]; r++){
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
    local_int_t * f2c_op
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // int num_fine_rows = nxf * nyf * nzf;
    local_int_t num_coarse_rows = static_cast<local_int_t>(nxc) * static_cast<local_int_t>(nyc) * static_cast<local_int_t>(nzc);

    for(local_int_t coarse_idx = tid; coarse_idx < num_coarse_rows; coarse_idx += blockDim.x * gridDim.x){
        int izc = coarse_idx / (nxc * nyc);
        int iyc = (coarse_idx % (nxc * nyc)) / nxc;
        int ixc = coarse_idx % nxc;

        int izf = izc * 2;
        int iyf = iyc * 2;
        int ixf = ixc * 2;

        local_int_t fine_idx = static_cast<local_int_t>(ixf) + static_cast<local_int_t>(nxf) * (iyf) + static_cast<local_int_t>(nxf) * static_cast<local_int_t>(nyf) * (izf);
        f2c_op[coarse_idx] = fine_idx;

        // if(coarse_idx < 5){
        //     printf("coarse_idx: %d, fine_idx: %d\n", coarse_idx, fine_idx);
        // }
    }
}

void generateHPCGProblem(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values,
    DataType * y    
) {

    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    get_num_elem_per_row_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr + 1);
    CHECK_CUDA(hipDeviceSynchronize());

    iterative_sum_kernel<<<1, 1>>>(row_ptr, num_rows);
    CHECK_CUDA(hipDeviceSynchronize());

    generateHPCGProblem_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr, col_idx, values, y);
    CHECK_CUDA(hipDeviceSynchronize());
   
}

void generateHPCGMatrix(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values
){
    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    get_num_elem_per_row_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr);
    CHECK_CUDA(hipDeviceSynchronize());

    iterative_sum_kernel<<<1, 1>>>(row_ptr, num_rows);
    CHECK_CUDA(hipDeviceSynchronize());

    // print the row_ptr
    // grab row_ptr from device
    // int * row_ptr_host = (int *) malloc((num_rows + 1) * sizeof(int));
    // CHECK_CUDA(hipMemcpy(row_ptr_host, row_ptr, (num_rows + 1) * sizeof(int), hipMemcpyDeviceToHost));

    // for (int i = 0; i < num_rows + 1; i++){
    //     std::cout << row_ptr_host[i] << " ";
    // }
    // std::cout << std::endl;

    generateHPCGMatrix_kernel<<<num_blocks, block_size>>>(nx, ny, nz, row_ptr, col_idx, values);
    CHECK_CUDA(hipDeviceSynchronize());
}

// we return how many non-zeros we ended up copying
local_int_t generate_striped_3D27P_Matrix_from_CSR(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values,
    int num_stripes, local_int_t * j_min_i,
    DataType * striped_A_d
){

    // allocate space for non-zeros
    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);
    local_int_t * num_nnz_i;
    CHECK_CUDA(hipMalloc(&num_nnz_i, num_rows * sizeof(local_int_t)));
    CHECK_CUDA(hipMemset(num_nnz_i, 0, num_rows * sizeof(local_int_t)));

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    generate_striped_from_CSR_kernel<<<num_blocks, block_size>>>(num_rows, num_stripes, j_min_i, row_ptr, col_idx, values, striped_A_d, num_nnz_i);

    CHECK_CUDA(hipDeviceSynchronize());
    
    // reduce using thrust
    thrust::device_ptr<local_int_t> num_nnz_i_ptr(num_nnz_i);
    local_int_t total_nnz = thrust::reduce(num_nnz_i_ptr, num_nnz_i_ptr + num_rows);

    // free memory
    CHECK_CUDA(hipFree(num_nnz_i));

    return total_nnz;

}

// we return how many non-zeros we ended up copying
local_int_t generate_CSR_from_Striped(
    local_int_t num_rows, int num_stripes,
    local_int_t * j_min_i, DataType * striped_A_d,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values
){
    // first we set the row_ptr
    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    // set the row_ptr first
    row_ptr_from_striped<<<num_blocks, block_size>>>(num_rows, num_stripes, j_min_i, striped_A_d, row_ptr);
    CHECK_CUDA(hipDeviceSynchronize());

    // we need to do the prefix sum
    iterative_sum_kernel<<<1, 1>>>(row_ptr, num_rows);
    CHECK_CUDA(hipDeviceSynchronize());

    // read the last element of row_ptr
    local_int_t num_nnz = 0;
    CHECK_CUDA(hipMemcpy(&num_nnz, row_ptr + num_rows, sizeof(local_int_t), hipMemcpyDeviceToHost));

    // now we generate the col_idx and values
    // std::cout << "j_min_i: " << j_min_i << std::endl;
    // std::cout << "num_rows: " << num_rows << std::endl;
    // std::cout << "num_stripes: " << num_stripes << std::endl;
    // std::cout << "striped_A_d: " << striped_A_d << std::endl;
    // std::cout << "row_ptr: " << row_ptr << std::endl;
    // std::cout << "col_idx: " << col_idx << std::endl;
    // std::cout << "values: " << values << std::endl;


    col_and_val_from_striped<<<num_blocks, block_size>>>(num_rows, num_stripes, j_min_i, striped_A_d, row_ptr, col_idx, values);
    CHECK_CUDA(hipDeviceSynchronize());

    return num_nnz;

}


void generate_f2c_operator(
    int nxc, int nyc, int nzc,
    int nxf, int nyf, int nzf,
    local_int_t * f2c_op
){

    local_int_t num_coarse_rows = static_cast<local_int_t>(nxc) * static_cast<local_int_t>(nyc) * static_cast<local_int_t>(nzc);

    int num_threads = 1024;
    int num_blocks = num_coarse_rows/num_threads;

    // we need at least 1 block
    num_blocks = num_blocks == 0 ? 1 : num_blocks;

    generate_f2c_operator_kernel<<<num_blocks, num_threads>>>(nxf, nyf, nzf, nxc, nyc, nzc, f2c_op);
    CHECK_CUDA(hipDeviceSynchronize());
}


__global__ void GenerateStripedPartialMatrix_kernel(int nx, int ny, int nz, global_int_t gnx, global_int_t gny, global_int_t gnz, global_int_t offset_x, global_int_t offset_y, global_int_t offset_z, DataType *A){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    local_int_t num_rows = nx * ny * nz;
    
    for (local_int_t i=tid; i<num_rows; i += blockDim.x * gridDim.x) {
        global_int_t gx = i % nx + offset_x;
        global_int_t gy = (i / nx) % ny + offset_y;
        global_int_t gz = i / (nx * ny) + offset_z;

        local_int_t id=i*27;
        for (int sz = -1; sz < 2; sz++){
            for(int sy = -1; sy < 2; sy++){
                for(int sx = -1; sx < 2; sx++){
                    if(gx + sx < 0 || gx + sx >= gnx ||
                        gy + sy < 0 || gy + sy >= gny ||
                        gz + sz < 0 || gz + sz >= gnz) {
                            A[id] = 0.0;
                    } 
                    else {
                        if(sx == 0 && sy == 0 && sz == 0){
                            A[id] = 26.0;
                        } 
                        else {
                            A[id] = -1.0;
                        }
                    }
                    id++;
                }
            }
        }



    }
}


__global__ void generate_partialf2c_operator_kernel(
    int nxf, int nyf, int nzf,
    int nxc, int nyc, int nzc,
    local_int_t * f2c_op_d
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // int num_fine_rows = nxf * nyf * nzf;
    local_int_t num_coarse_rows = nxc * nyc * nzc;

    for(local_int_t coarse_idx = tid; coarse_idx < num_coarse_rows; coarse_idx += blockDim.x * gridDim.x){
        int izc = coarse_idx / (nxc * nyc);
        int iyc = (coarse_idx % (nxc * nyc)) / nxc;
        int ixc = coarse_idx % nxc;

        int izf = izc * 2;
        int iyf = iyc * 2;
        int ixf = ixc * 2;

        local_int_t fine_idx = ixf + nxf * iyf + nxf * nyf * izf;
        f2c_op_d[coarse_idx] = fine_idx;

        // if(coarse_idx < 5){
        //     printf("coarse_idx: %d, fine_idx: %d\n", coarse_idx, fine_idx);
        // }
    }
}


void GenerateStripedPartialMatrix_GPU(Problem *problem, DataType *A_d) {
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    local_int_t num_rows = nx * ny * nz;

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    GenerateStripedPartialMatrix_kernel<<<num_blocks, block_size>>>(problem->nx, problem->ny, problem->nz, problem->gnx, problem->gny, problem->gnz, problem->gx0, problem->gy0, problem->gz0, A_d);
}

void generate_partialf2c_operator(
    int nxc, int nyc, int nzc,
    int nxf, int nyf, int nzf,
    local_int_t * f2c_op_d
){

    local_int_t num_coarse_rows = nxc * nyc * nzc;

    int num_threads = 1024;
    int num_blocks = (num_coarse_rows+num_threads-1)/num_threads;

    generate_partialf2c_operator_kernel<<<num_blocks, num_threads>>>(nxf, nyf, nzf, nxc, nyc, nzc, f2c_op_d);
    //CHECK_CUDA(hipDeviceSynchronize());
}


__global__ void generate_y_vector_for_HPCG_problem_kernel(global_int_t gnx, global_int_t gny, global_int_t gnz, int nx, int ny, int nz, global_int_t gx0, global_int_t gy0, global_int_t gz0, DataType *y){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (local_int_t i = tid; i < nx*ny*nz; i += blockDim.x * gridDim.x){
        global_int_t iz = i / (nx*ny) + gz0;
        global_int_t iy = i % (nx*ny) / nx + gy0;
        global_int_t ix = i - iz * (nx*ny) - iy * nx + gx0;

        local_int_t nnz_i=0;

        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < gnz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < gny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < gnx){
                                /* int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz); */
                                nnz_i++;
                            }
                        }
                    }
                }
            }
        }
        y[i] = 26.0 - nnz_i;
    }

}

void generate_y_vector_for_HPCG_problem_onGPU(Problem *problem, DataType*y_d){
    local_int_t num_rows = problem->nx * problem->ny * problem->nz;
    local_int_t num_cols = problem->nx * problem->ny * problem->nz;

    int nthread=256;
    local_int_t nblocks=(num_rows+nthread-1) / nthread;
    generate_y_vector_for_HPCG_problem_kernel<<<nblocks, nthread>>>(problem->gnx, problem->gny, problem->gnz, problem->nx, problem->ny, problem->nz, problem->gx0, problem->gy0, problem->gz0, y_d);
}

