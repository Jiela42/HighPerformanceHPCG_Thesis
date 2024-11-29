#include "HPCG_versions/banded_warp_reduction.cuh"

#include <cuda_runtime.h>
 
 __global__ void banded_warp_reduction_SPMV_kernel(
            double* banded_A,
            int num_rows, int num_bands, int * j_min_i,
            double* x, double* y
        ){
            
            int cooperation_number = 4;
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int lane = threadIdx.x % cooperation_number;

            // every thread computes one or more rows of the matrix
            for (int i = tid/cooperation_number; i < num_rows; i += (blockDim.x * gridDim.x)/cooperation_number) {
                // compute the matrix-vector product for the ith row
                double sum_i = 0;
                for (int band = lane; band < num_bands; band += cooperation_number) {
                    int j = i + j_min_i[band];
                    int current_row = i * num_bands;
                    if (j >= 0 && j < num_rows) {
                        sum_i += banded_A[current_row + band] * x[j];
                    }
                }

                // now let's reduce the sum_i to a single value using warp-level reduction
                for(int offset = cooperation_number/2; offset > 0; offset /= 2){
                    sum_i += __shfl_down_sync(0xFFFFFFFF, sum_i, offset);
                }

                if (lane == 0){
                    y[i] = sum_i;
                }
            }
        }
    