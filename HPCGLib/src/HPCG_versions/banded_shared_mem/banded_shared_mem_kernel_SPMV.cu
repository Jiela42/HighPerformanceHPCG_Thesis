#include "HPCG_versions/naiveBanded.cuh"

#include <cuda_runtime.h>
 
__device__ void print_values_cooperatively(double * array, int num_elements){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_elements; i += blockDim.x * gridDim.x){
        printf("array[%d]: %f\n", i, array[i]);
    }
}

__device__ void print_values_thread(double * array, int num_elements){

    for (int i = 0; i < num_elements; i++){
        printf("array[%d]: %f\n", i, array[i]);
    }
}

__device__ void print_offsets_thread(int * array, int num_elements){

    for (int i = 0; i < num_elements; i++){
        printf("array[%d]: %d\n", i, array[i]);
    }
}

__device__ void test_val_cooperatively(double * array, int num_elements, double test_val){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_elements; i += blockDim.x * gridDim.x){
       if(array[i] != test_val){
           printf("array[%d]: %f\n", i, array[i]);
       }
       
    }
}

 __global__ void banded_shared_memory_SPMV_kernel(
            int rows_per_sm, int num_x_elem, int num_consecutive_memory_regions,
            int* min_j, int* max_j,
            double* banded_A,
            int num_rows, int num_bands, int * j_min_i,
            double* x, double* y
        ){
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int necessary_pad = 0;
            if((num_bands + 3*num_consecutive_memory_regions) %2 != 0){
                necessary_pad = 1;
            }
            extern __shared__ int shared_mem[];
            int* shared_j_min_i = shared_mem;
            int* shared_j_min = (int*)& shared_j_min_i[num_bands];
            int* shared_j_max = (int*)& shared_j_min[num_consecutive_memory_regions];
            int * sum_x_elem_per_conseq_mem = (int*)& shared_j_max[num_consecutive_memory_regions];
            double* shared_x = (double*)& sum_x_elem_per_conseq_mem[num_consecutive_memory_regions + necessary_pad];  

            // int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int bid = blockIdx.x * blockDim.x;
            int total_size = blockDim.x * gridDim.x;

            int sum_x_elem = 0;

            // first the first thread of each block loads any offsets

            if (threadIdx.x == 0){
            // printf("shard_x before\n");
            // print_values_thread(shared_x, 8);

            // print the pointer addresses
            // printf("shared_j_min_i: %p\n", shared_j_min_i);
            // printf("shared_j_min: %p\n", shared_j_min);
            // printf("shared_j_max: %p\n", shared_j_max);
            // printf("sum_x_elem_per_conseq_mem: %p\n", sum_x_elem_per_conseq_mem);
            // printf("shared_x: %p\n", shared_x);
            // printf("banded_A: %p\n", banded_A);
            // printf("x: %p\n", x);

            // print_offsets_thread(shared_j_min, 8);
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

                // print_offsets_thread(shared_j_min_i, 8);
                if (tid == 0){

                printf("num_x_elem: %d\n", num_x_elem);
                printf("num_consecutive_memory_regions: %d\n", num_consecutive_memory_regions);
                printf("j_min\n");
                print_offsets_thread(shared_j_min, 1);
                printf("j_max\n");
                print_offsets_thread(shared_j_max, 1);
                printf("sum_x_elem_per_conseq_mem\n");
                print_offsets_thread(sum_x_elem_per_conseq_mem, 2);
                printf("sum_x_elem: %d\n", sum_x_elem);
                printf("num_threads_in block: %d\n", blockDim.x);
                }
            }

            // synchronize the threads after preliminary loading
            __syncthreads();

            // every thread computes one or more rows of the matrix
            for (int iter = 0; iter*total_size < num_rows; iter++) {
                // does this not do anything? or does it just copy the wrong values?

                // row_start refers to the first row of this SM
                int row_start = iter * total_size + bid;

                // between each block of rows of the matrix, we need to allocate the new entries of x
                // these entries of x depend on the row start
                // now we actually load x

                for (int i = threadIdx.x; i < num_x_elem; i += blockDim.x){


                    // first we find which offset we need to take
                    for (int j = 0; j < num_consecutive_memory_regions; j++){
                    if(tid ==0){
                        printf("block_idx: %d\n", blockIdx.x);
                        printf("i: %d\n", i);
                        printf("j: %d\n", j);
                        printf("sum_x_elem_per_conseq_mem ptr: %p\n", sum_x_elem_per_conseq_mem);
                        printf("sum_x_elem_per_conseq_mem[j]: %d\n", sum_x_elem_per_conseq_mem[j]);
                        printf("sum_x_elem_per_conseq_mem[j+1]: %d\n", sum_x_elem_per_conseq_mem[j+1]);
                    }
                        if (i >= sum_x_elem_per_conseq_mem[j] && i < sum_x_elem_per_conseq_mem[j+1]){

                            // so the first x_element to load is a j and it is j_min[0]+row_start
                            // then we get first_elem + 1, ..., j_max[0] + row_start
                            // so what is the i-th elem?
                            // it is j_min_i[j] + i + row_start
                            int target_j = shared_j_min[j] + i + row_start;
                            // if(threadIdx.x==0){
                            //     printf("target_j: %d\n", target_j);
                            // }
                            // check that it is within the bounds of x
                            if(target_j >= 0 && target_j < num_rows){
                                shared_x[i] = x[target_j];
                                // if(i < 8){
                                //     printf("shared_x[%d]: %f\n", i, shared_x[i]);
                                // }
                                // printf("printing from written x\n");
                                // printf("shared_x[%d]: %f\n", i, shared_x[i]);
                                // printf("x[%d]: %f\n", target_j, x[target_j]);
                            }

                            // once we found it, we break
                            break;
                        }
                    }
                }

            

                // synchronize the threads after loading x
                __syncthreads();

                test_val_cooperatively(shared_x, 513, 1.0);

                // if(threadIdx.x == 0)
                //     {printf("shared_x after\n");
                //     printf("shared_x[0]: %f\n", shared_x[0]);
                //     // printf()
                // }
                // print_values_cooperatively(shared_x, 8);
                // if(threadIdx.x ==0){

                // for(int meow = 0; meow < num_bands; meow++){
                //     double val = banded_A[meow];
                //     if (val != 0.0){
                //         printf("banded_A[%d]: %f\n", meow, val);
                //     }
                // }
                // }
                
                // now we compute the matrix-vector product
                for(int row = row_start + threadIdx.x; row < row_start + rows_per_sm && row < num_rows; row += blockDim.x){
                    // compute the matrix-vector product for the ith row
                    if(row == 0){
                        printf("yi before: %f\n", y[row]);
                    }
                    double sum_i = 0;
                    for (int band = 0; band < num_bands; band++) {
                        int j = row + shared_j_min_i[band];
                        int current_row = row * num_bands;
                        if (j >= 0 && j < num_rows) {
                            sum_i += banded_A[current_row + band] * shared_x[j];
                            // if (row == 0){
                            //     printf("banded_A: %f\n", banded_A[current_row + band]);
                            //     printf("shared_x: %f\n", shared_x[j]);
                            // }
                        }
                    }
                    y[row] = sum_i;
                    if(row == 0){
                    printf("yi after: %f\n", y[row]);
                    }
                }
            }
        }
    