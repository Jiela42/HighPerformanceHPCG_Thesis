#include "HPCG_versions/striped_shared_mem.cuh"
#include "UtilLib/utils.cuh"
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

__global__ void striped_shared_memory_SPMV_kernel(
        int rows_per_sm, int num_x_elem, int num_consecutive_memory_regions,
        int* min_j, int* max_j,
        double* striped_A,
        int num_rows, int num_stripes, int * j_min_i,
        double* x, double* y
    )
{
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int bid = blockIdx.x * blockDim.x;
    int total_size_of_worked_on_rows = rows_per_sm * gridDim.x;

    int necessary_pad = 0;
    if((num_stripes + 3*num_consecutive_memory_regions + 1) %2 != 0){
        necessary_pad = 1;
    }
    extern __shared__ int shared_mem[];
    int* shared_j_min_i = shared_mem;
    int* shared_j_min = (int*)& shared_j_min_i[num_stripes];
    int* shared_j_max = (int*)& shared_j_min[num_consecutive_memory_regions];
    // we need to explain why the sum_x_elem_per_conseq_mem is +1 -> basically the same reason why we get num_rows+1 many rowpointers in csr format
    int * sum_x_elem_per_conseq_mem = (int*)& shared_j_max[num_consecutive_memory_regions];
    double* shared_x = (double*)& sum_x_elem_per_conseq_mem[num_consecutive_memory_regions + 1 + necessary_pad];  

    int sum_x_elem = 0;

    // first the first thread of each block loads any offsets
    if (threadIdx.x == 0){

        for (int i = 0; i < num_stripes; i++){
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

        // between each block of rows of the matrix, we need to allocate the new entries of x
        // these entries of x depend on the row start
        // now we actually load x

        for (int i = threadIdx.x; i < num_x_elem; i += blockDim.x){

            // first we find which offset we need to take
            for (int j = 0; j < num_consecutive_memory_regions; j++){

                if (i >= sum_x_elem_per_conseq_mem[j] && i < sum_x_elem_per_conseq_mem[j+1]){

                    // so the first x_element to load is a j and it is j_min[0]+row_start
                    // then we get first_elem + 1, ..., j_max[0] + row_start
                    // so what is the i-th elem?
                    // it is j_min_i[j] + i + row_start
                    int target_j = shared_j_min[j] + i + row_start - sum_x_elem_per_conseq_mem[j];

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
            for (int stripe = 0; stripe < num_stripes; stripe++) {
                int j = row + shared_j_min_i[stripe];
                int current_row = row * num_stripes;

                for(int mem_reg = 0; mem_reg < num_consecutive_memory_regions; mem_reg++){
                    // we test if it is in a specific memory segment, which is also dependent on the row_start
                    if(j >= shared_j_min[mem_reg] + row_start && j < shared_j_max[mem_reg] + row_start){
                        j = j - (shared_j_min[mem_reg] + row_start) + sum_x_elem_per_conseq_mem[mem_reg];
                        break;
                    }
                }
                
                if (j >= 0 && j < num_rows) {
                    // we have to calculate what j would be. since we only have 1410 rows per block but j can be up to num_rows
                    
                    sum_i += striped_A[current_row + stripe] * shared_x[j];
                }
            }
            y[row] = sum_i;
        }
    }
}
    


void eliminate_overlap(int* min_j, int* max_j, int num_thick_stripes, int* num_x_elem, int* num_consecutive_memory_regions){

    int ctr_consecutive_memory_regions = 1;
    // int nx = 0;
    *num_x_elem = 0;

    for (int i = 1; i < num_thick_stripes; i++){
        if (max_j[ctr_consecutive_memory_regions-1] >= min_j[i]){
            max_j[ctr_consecutive_memory_regions-1] = max_j[i];
        }
        else{
            min_j[ctr_consecutive_memory_regions] = min_j[i];
            max_j[ctr_consecutive_memory_regions] = max_j[i];
            ctr_consecutive_memory_regions ++;
        }
    }

    for (int i = 0; i < ctr_consecutive_memory_regions; i++){
        *num_x_elem += max_j[i] - min_j[i];
    }

    *num_consecutive_memory_regions = ctr_consecutive_memory_regions;
}

template <typename T>
void Striped_Shared_Memory_Implementation<T>::striped_shared_memory_computeSPMV(
        striped_Matrix<T>& A, //we only pass A for the metadata
        T * striped_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the striped matrix
        int num_stripes, // the number of stripes in the striped matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

        // since every thread is working on one or more rows we need to base the number of threads on that

        // dynamically calculate how many rows can be handled at the same time
        // i.e. how many doubles are required per row

        std::vector<int> j_min_i_host = A.get_j_min_i();

        int new_elem_per_row = 1;

        for (int i = 1; i < num_stripes; i++){
            if (j_min_i_host[i-1] + 1 != j_min_i_host[i]){
                new_elem_per_row ++;
            }
        }

        int shared_mem_bytes = 1024 * SHARED_MEM_SIZE;
        int shared_mem_doubles = shared_mem_bytes / sizeof(double);
        int shared_mem_doubles_for_x = shared_mem_doubles - 2*num_stripes;

        int rows_per_sm = (shared_mem_doubles_for_x -  2 * num_stripes) / new_elem_per_row;
        // rows_per_sm = next_smaller_power_of_two(rows_per_sm);
        int num_threads = 1024;
        int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, rows_per_sm));
        int min_j [num_stripes];
        int max_j [num_stripes];
        int num_thick_stripes = 1;
        int num_conseq_mem_reg = 0;
        int num_x_elem = 0;
                
        if(shared_mem_doubles - num_stripes - 2 > num_rows){
            rows_per_sm = num_rows;
            min_j[0] = 0;
            max_j[0] = num_rows;
            num_conseq_mem_reg = 1;
            num_x_elem = num_rows;
        }
        else{

            // based on the guess of the number of rows per sm we calculate the exact numbers            
            // in the kernel we will need to adjust this, by adding the row start
            // we also need to adjust this such that we don't pass in negative values
            min_j[0] = j_min_i_host[0];
            max_j[0] = j_min_i_host[0] + rows_per_sm;

            // this refers to how many independent stripes we end up having
            for (int stripe = 1; stripe < num_stripes; stripe++){
                
                if (j_min_i_host[stripe-1] + 1 == j_min_i_host[stripe]){
                    max_j[num_thick_stripes-1] += 1;
                }
                else{
                    // again in the kernel this will need to be adjusted
                    min_j[num_thick_stripes] = j_min_i_host[stripe];
                    max_j[num_thick_stripes] = j_min_i_host[stripe] + rows_per_sm;
                    num_thick_stripes ++;
                }
            }
            eliminate_overlap(min_j, max_j, num_thick_stripes, &num_x_elem, &num_conseq_mem_reg);
        }


        int size_shared_j_min_i = num_stripes * sizeof(int);
        int size_x_offsets = 3*num_conseq_mem_reg * sizeof(int);
        int size_shared_x = num_x_elem * sizeof(double);
        int size_shared_memory = size_shared_j_min_i  + size_x_offsets + size_shared_x;

        assert(size_shared_memory < shared_mem_bytes);
        // move the offsets to the device
        // I am not sure if this is actually faster,
        // or if it would make sense to have the first thread in a block do this computation
        int * min_j_d;
        int * max_j_d;

        CHECK_CUDA(cudaMalloc(&min_j_d, num_conseq_mem_reg * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&max_j_d, num_conseq_mem_reg * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(min_j_d, min_j, num_conseq_mem_reg * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(max_j_d, max_j, num_conseq_mem_reg * sizeof(int), cudaMemcpyHostToDevice)); 


        // call the kernel
        striped_shared_memory_SPMV_kernel<<<num_blocks, num_threads, size_shared_memory>>>(
            rows_per_sm, num_x_elem, num_conseq_mem_reg,
            min_j_d, max_j_d,
            striped_A_d, num_rows, num_stripes, j_min_i,
            x_d, y_d
        );

        // synchronize the device
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

// explicit template instantiation
template class Striped_Shared_Memory_Implementation<double>;