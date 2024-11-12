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
    int bid = blockIdx.x * blockDim.x;
    int total_size_of_worked_on_rows = rows_per_sm * gridDim.x;

    int necessary_pad = 0;
    if((num_bands + 3*num_consecutive_memory_regions + 1) %2 != 0){
        necessary_pad = 1;
    }
    extern __shared__ int shared_mem[];
    int* shared_j_min_i = shared_mem;
    int* shared_j_min = (int*)& shared_j_min_i[num_bands];
    int* shared_j_max = (int*)& shared_j_min[num_consecutive_memory_regions];
    // we need to explain why the sum_x_elem_per_conseq_mem is +1 -> basically the same reason why we get num_rows+1 many rowpointers in csr format
    int * sum_x_elem_per_conseq_mem = (int*)& shared_j_max[num_consecutive_memory_regions];
    double* shared_x = (double*)& sum_x_elem_per_conseq_mem[num_consecutive_memory_regions + 1 + necessary_pad];  

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

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


    // every thread computes one or more rows of the matrix
    for (int iter = 0; iter*total_size_of_worked_on_rows < num_rows; iter++) {
        // synchronize the threads after loading the offsets and after each iteration
        __syncthreads();

        // row_start refers to the first row of this SM
        // which depends on the number of rows per sm, the iteration and the block_id
        int row_start = iter * total_size_of_worked_on_rows + blockIdx.x * rows_per_sm;

        // if(blockIdx.x == 181 && threadIdx.x == 0){
        //     printf("blockIdx: %d\n", blockIdx.x);
        //     printf("row_start: %d\n", row_start);
        //     for(int meow = 0; meow < num_consecutive_memory_regions; meow++){
        //         printf("min_j[%d]: %d max_j[%d]: %d\n", meow, min_j[meow] + row_start, meow, max_j[meow]+ row_start);
        //     }
        // }

        // between each block of rows of the matrix, we need to allocate the new entries of x
        // these entries of x depend on the row start
        // now we actually load x

        for (int i = threadIdx.x; i < num_x_elem; i += blockDim.x){

            // first we find which offset we need to take
            for (int j = 0; j < num_consecutive_memory_regions; j++){

            // if(i==3144 && blockIdx.x == 181 && row_start == 255210){
            //     printf("i: %d, j: %d, sum_x_elem_per_conseq_mem[j]: %d, sum_x_elem_per_conseq_mem[j+1]: %d\n", i, j, sum_x_elem_per_conseq_mem[j], sum_x_elem_per_conseq_mem[j+1]);
            // }
                if (i >= sum_x_elem_per_conseq_mem[j] && i < sum_x_elem_per_conseq_mem[j+1]){

                    // so the first x_element to load is a j and it is j_min[0]+row_start
                    // then we get first_elem + 1, ..., j_max[0] + row_start
                    // so what is the i-th elem?
                    // it is j_min_i[j] + i + row_start
                    int target_j = shared_j_min[j] + i + row_start - sum_x_elem_per_conseq_mem[j];

                    // if(i==3144 && blockIdx.x == 181 && row_start == 255210){
                    //     printf("bid: %d, target_j: %d\n",blockIdx.x, target_j);
                    // }

                    // check that it is within the bounds of x
                    if(target_j >= 0 && target_j < num_rows){
                        // printf("target_j: %d, blockid %d, threadid: %d\n", target_j, blockIdx.x, threadIdx.x);
                        shared_x[i] = x[target_j];
                        // if(target_j == 259306){
                        //     printf("target_j: %d, i: %d, blockid %d, threadid: %d\n", target_j,i, blockIdx.x, threadIdx.x);
                        // }
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
            // if(row == 255210){
            //     printf("row: %d, bid: %d, tid: %d, min_j %d, max_j %d\n", row, blockIdx.x, threadIdx.x, min_j[2] + row_start, max_j[2]+ row_start);
            // }
            for (int band = 0; band < num_bands; band++) {
                int j = row + shared_j_min_i[band];
                int current_row = row * num_bands;
                
                // if(row == 255210){
                //     printf("band: %d, j: %d\n", band, j);
                // }


                // printf("reading row: %d from bid %d and thread number %d \n", row, blockIdx.x, threadIdx.x);
                // debug notes: it looks like we tried to read all the rows
                // we iterate over the consecutive memory regions to find the correct j
                for(int mem_reg = 0; mem_reg < num_consecutive_memory_regions; mem_reg++){
                    // we test if it is in a specific memory segment, which is also dependent on the row_start
                    if(j >= shared_j_min[mem_reg] + row_start && j < shared_j_max[mem_reg] + row_start){
                        j = j - (shared_j_min[mem_reg] + row_start) + sum_x_elem_per_conseq_mem[mem_reg];
                        break;
                    }
                }

                // if (row == 255210)
                // {
                //     printf("new j: %d\n", j);
                // }
                
                if (j >= 0 && j < num_rows) {
                    // we have to calculate what j would be. since we only have 1410 rows per block but j can be up to num_rows
                    
                    sum_i += banded_A[current_row + band] * shared_x[j];
                    // if (row == 255210){
                    //     printf("banded_A: %f\n", banded_A[current_row + band]);
                    //     printf("shared_x: %f\n", shared_x[j]);
                    // }
                }
            }
            // printf("writing row: %d from bid %d and thread number %d \n", row, blockIdx.x, threadIdx.x);
            y[row] = sum_i;

        }
    }
}
    