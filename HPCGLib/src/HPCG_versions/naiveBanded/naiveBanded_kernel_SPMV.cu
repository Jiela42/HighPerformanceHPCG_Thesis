#include "HPCG_versions/naiveBanded.cuh"

#include <cuda_runtime.h>
 
 __global__ void naiveBanded_SPMV_kernel(
            double* banded_A,
            int num_rows, int num_bands, int * j_min_i,
            double* x, double* y
        ){

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // every thread computes one or more rows of the matrix

            for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
                // compute the matrix-vector product for the ith row
                int sum_i = 0;
                for (int band = 0; band < num_bands; band++) {
                    int j = i + j_min_i[band];
                    int current_row = i * num_rows;
                    if (j >= 0 && j < num_bands; j++) {
                        sum_i += banded_A[current_row + band] * x[j];
                    }
                }
                y[i] = sum_i;
            }
        }