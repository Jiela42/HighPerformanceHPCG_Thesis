#include "HPCG_versions/naiveBanded.cuh"

#include <cuda_runtime.h>
 
 __global__ void banded_shared_memory_SPMV_kernel(
            int rows_per_sm
            double* banded_A,
            int num_rows, int num_bands, int * j_min_i,
            double* x, double* y
        ){
            // shared_j_min_i
            // shared_x
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int bid = blockIdx.x * blockDim.x;
            int total_size = blockDim.x * gridDim.x;

            // every thread computes one or more rows of the matrix
            for (int iter = 0; iter*total_size < num_rows; iter++) {

                // between each row of the matrix, we need to allocate the new entries of x
                int [num_bands] min_j;
                int [num_bands] max_j;
                int [num_bands] consecutive_j;
                // row_start refers to the first row of this SM
                int row_start = iter * total_size + bid;

                min_j[0] = j_min_i[0] - row_start;
                max_j[0] = j_min_i[0] - (row_start + num_rows_per_sm);
                consecutive_j[0] = num_rows_per_sm;


                // this refers to how many independent bands we end up having
                int num_thick_bands = 1;
                

                for (int band = 1; band < num_bands; band++){
                    
                    if (j_min_i[band-1] + 1 == j_min_i[band]){
                        max_j[num_thick_bands-1] += 1;
                        consecutive_j[num_thick_bands-1] += 1
                    }
                    else{
                        min_j[num_thick_bands] = j_min_i[band] - row_start;
                        max_j[num_thick_bands] = j_min_i[band] - (row_start + num_rows_per_sm);
                        consecutive_j[num_thick_bands] = num_rows_per_sm;
                        num_thick_bands ++;
                    }
                }
                
                int num_x_elem = 0;
                for(int i = 0; i < num_thick_bands; i++){
                    num_x_elem += consecutive_j[i];
                }

                // now we actually load x

                for (int i = threadIdx.x; i < num_x_elem; I += blockDim){

                }


                int row = row_start + threadIdx.x
                if(row < num_rows){
                    // compute the matrix-vector product for the ith row
                    double sum_i = 0;
                    for (int band = 0; band < num_bands; band++) {
                        int j = i + j_min_i[band];
                        int current_row = i * num_bands;
                        if (j >= 0 && j < num_rows) {
                            sum_i += banded_A[current_row + band] * x[j];
                        }
                    }
                    y[i] = sum_i;
                }
            }
        }
    