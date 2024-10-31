#include "HPCG_versions/naiveBanded.cuh"

#include <cuda_runtime.h>
 
 __global__ void banded_shared_memory_SPMV_kernel(
            int rows_per_sm, int num_x_elem, int num_consecutive_memory_regions,
            int* min_j, int* max_j,
            double* banded_A,
            int num_rows, int num_bands, int * j_min_i,
            double* x, double* y
        ){

            extern __shared__ int shared_mem[];
            int* shared_j_min_i = shared_mem;
            int* shared_j_min = (int*)& shared_j_min_i[num_bands];
            int* shared_j_max = (int*)& shared_j_min[num_consecutive_memory_regions];
            int * sum_x_elem_per_conseq_mem = (int*)& shared_j_max[num_consecutive_memory_regions];
            double* shared_x = (double*)& sum_x_elem_per_conseq_mem[num_consecutive_memory_regions + 1];  

            // int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int bid = blockIdx.x * blockDim.x;
            int total_size = blockDim.x * gridDim.x;

            int sum_x_elem = 0;

            // first the first thread of each block loads any offsets

            if (threadIdx.x == 0){
                for (int i = 0; i < num_bands; i++){
                    shared_j_min_i[i] = j_min_i[i];
                }
                for (int i = 0; i < num_consecutive_memory_regions; i++){
                    shared_j_min[i] = min_j[i];
                    shared_j_max[i] = max_j[i];
                    sum_x_elem_per_conseq_mem[i] = sum_x_elem;
                    sum_x_elem += max_j[i] - min_j[i];
                }
                // we need to also have the last element of sum_x_elem_per_conseq_mem (where all of them are summed up i.e. num_x_elem)
                sum_x_elem_per_conseq_mem[num_consecutive_memory_regions] = sum_x_elem;
            }

            // synchronize the threads after preliminary loading
            __syncthreads();

            // every thread computes one or more rows of the matrix
            for (int iter = 0; iter*total_size < num_rows; iter++) {

                // row_start refers to the first row of this SM
                int row_start = iter * total_size + bid;

                // between each block of rows of the matrix, we need to allocate the new entries of x
                // these entries of x depend on the row start
                // now we actually load x

                for (int i = threadIdx.x; i < num_x_elem; i += blockDim.x){

                    // first we find which offset we need to take
                    for (int j = 0; j < num_consecutive_memory_regions; j++){
                        if (i < sum_x_elem_per_conseq_mem[j] && i < sum_x_elem_per_conseq_mem[j+1]){

                            // so the first x_element to load is a j and it is j_min[0]+row_start
                            // then we get first_elem + 1, ..., j_max[0] + row_start
                            // so what is the i-th elem?
                            // it is j_min[j] + i + row_start
                            int target_j = shared_j_min[j] + i + row_start;
                            // check that it is within the bounds of x
                            if(target_j >= 0 && target_j < num_rows){
                                shared_x[i] = x[target_j];
                            }

                            // once we found it, we break
                            break;
                        }
                    }
                }

                // synchronize the threads after loading x
                __syncthreads();

                // now we compute the matrix-vector product
                for(int row = row_start + threadIdx.x; row < row_start + rows_per_sm && row < num_rows; row += blockDim.x){
                    // compute the matrix-vector product for the ith row
                    double sum_i = 0;
                    for (int band = 0; band < num_bands; band++) {
                        int j = row + shared_j_min_i[band];
                        int current_row = row * num_bands;
                        if (j >= 0 && j < num_rows) {
                            sum_i += banded_A[current_row + band] * shared_x[j];
                        }
                    }
                    y[row] = sum_i;
                }
            }
        }
    