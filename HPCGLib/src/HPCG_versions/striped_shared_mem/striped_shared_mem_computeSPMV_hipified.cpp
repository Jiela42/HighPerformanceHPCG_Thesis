#include "hip/hip_runtime.h"
#include "HPCG_versions/striped_shared_mem_hipified.cuh"
#include "UtilLib/utils_hipified.cuh"
#include <hip/hip_runtime.h>

 
__device__ void print_values_cooperatively(DataType * array, local_int_t num_elements){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (local_int_t i = tid; i < num_elements; i += blockDim.x * gridDim.x){
        printf("array[%ld]: %f\n", i, array[i]);
    }
}

__device__ void print_values_thread(DataType * array, local_int_t num_elements){

    for (local_int_t i = 0; i < num_elements; i++){
        printf("array[%ld]: %f\n", i, array[i]);
    }
}

__device__ void print_offsets_thread(int * array, local_int_t num_elements){

    for (local_int_t i = 0; i < num_elements; i++){
        printf("array[%ld]: %d\n", i, array[i]);
    }
}

__device__ void test_val_cooperatively(DataType * array, local_int_t num_elements, DataType test_val){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (local_int_t i = tid; i < num_elements; i += blockDim.x * gridDim.x){
       if(array[i] != test_val){
           printf("array[%ld]: %f\n", i, array[i]);
       }
       
    }
}

__global__ void striped_shared_memory_SPMV_kernel(
        local_int_t rows_per_sm, local_int_t num_x_elem, int num_consecutive_memory_regions,
        local_int_t* min_j, local_int_t* max_j,
        DataType* striped_A,
        local_int_t num_rows, int num_stripes, local_int_t * j_min_i,
        DataType* x, DataType* y
    )
{
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int bid = blockIdx.x * blockDim.x;
    int total_size_of_worked_on_rows = rows_per_sm * gridDim.x;

    int necessary_pad = 0;
    if((num_stripes + 3*num_consecutive_memory_regions + 1) %2 != 0){
        necessary_pad = 1;
    }
    extern __shared__ local_int_t shared_mem[];
    local_int_t* shared_j_min_i = shared_mem;
    local_int_t* shared_j_min = (local_int_t*)& shared_j_min_i[num_stripes];
    local_int_t* shared_j_max = (local_int_t*)& shared_j_min[num_consecutive_memory_regions];
    // we need to explain why the sum_x_elem_per_conseq_mem is +1 -> basically the same reason why we get num_rows+1 many rowpointers in csr format
    local_int_t * sum_x_elem_per_conseq_mem = (local_int_t*)& shared_j_max[num_consecutive_memory_regions];
    DataType* shared_x = (DataType*)& sum_x_elem_per_conseq_mem[num_consecutive_memory_regions + 1 + necessary_pad];  

    local_int_t sum_x_elem = 0;

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
    for (local_int_t iter = 0; iter*total_size_of_worked_on_rows < num_rows; iter++) {
        // synchronize the threads after loading the offsets and after each iteration
        __syncthreads();

        // row_start refers to the first row of this SM
        // which depends on the number of rows per sm, the iteration and the block_id
        local_int_t row_start = iter * total_size_of_worked_on_rows + blockIdx.x * rows_per_sm;

        // between each block of rows of the matrix, we need to allocate the new entries of x
        // these entries of x depend on the row start
        // now we actually load x

        for (local_int_t i = threadIdx.x; i < num_x_elem; i += blockDim.x){

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
        for(local_int_t row = row_start + threadIdx.x; row < row_start + rows_per_sm && row < num_rows; row += blockDim.x){
            // compute the matrix-vector product for the ith row
            DataType sum_i = 0;
            for (int stripe = 0; stripe < num_stripes; stripe++) {
                local_int_t j = row + shared_j_min_i[stripe];
                local_int_t current_row = row * num_stripes;

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
    


void eliminate_overlap(local_int_t* min_j, local_int_t* max_j, int num_thick_stripes, local_int_t* num_x_elem, int* num_consecutive_memory_regions){

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
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

        // since every thread is working on one or more rows we need to base the number of threads on that

        // dynamically calculate how many rows can be handled at the same time
        // i.e. how many DataTypes are required per row

        local_int_t num_rows = A.get_num_rows();
        int num_stripes = A.get_num_stripes();
        local_int_t * j_min_i = A.get_j_min_i_d();
        T * striped_A_d = A.get_values_d();

        std::vector<local_int_t> j_min_i_host(num_stripes);

        // copy j_min_i to the host
        CHECK_CUDA(hipMemcpy(j_min_i_host.data(), j_min_i, num_stripes * sizeof(local_int_t), hipMemcpyDeviceToHost));

        int new_elem_per_row = 1;

        for (int i = 1; i < num_stripes; i++){
            if (j_min_i_host[i-1] + 1 != j_min_i_host[i]){
                new_elem_per_row ++;
            }
        }

        int shared_mem_bytes = 1024 * SHARED_MEM_SIZE;
        int shared_mem_doubles = shared_mem_bytes / sizeof(DataType);
        int shared_mem_doubles_for_x = shared_mem_doubles - 2*num_stripes;

        local_int_t rows_per_sm = (shared_mem_doubles_for_x -  2 * num_stripes) / new_elem_per_row;
        // rows_per_sm = next_smaller_power_of_two(rows_per_sm);
        int num_threads = 1024;
        int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(num_rows, rows_per_sm));
        local_int_t min_j [num_stripes];
        local_int_t max_j [num_stripes];
        int num_thick_stripes = 1;
        int num_conseq_mem_reg = 0;
        local_int_t num_x_elem = 0;
                
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


        local_int_t size_shared_j_min_i = num_stripes * sizeof(local_int_t);
        local_int_t size_x_offsets = 3*num_conseq_mem_reg * sizeof(local_int_t);
        local_int_t size_shared_x = num_x_elem * sizeof(DataType);
        local_int_t size_shared_memory = size_shared_j_min_i  + size_x_offsets + size_shared_x;

        assert(size_shared_memory < shared_mem_bytes);
        // move the offsets to the device
        // I am not sure if this is actually faster,
        // or if it would make sense to have the first thread in a block do this computation
        local_int_t * min_j_d;
        local_int_t * max_j_d;

        CHECK_CUDA(hipMalloc(&min_j_d, num_conseq_mem_reg * sizeof(local_int_t)));
        CHECK_CUDA(hipMalloc(&max_j_d, num_conseq_mem_reg * sizeof(local_int_t)));

        CHECK_CUDA(hipMemcpy(min_j_d, min_j, num_conseq_mem_reg * sizeof(local_int_t), hipMemcpyHostToDevice));
        CHECK_CUDA(hipMemcpy(max_j_d, max_j, num_conseq_mem_reg * sizeof(local_int_t), hipMemcpyHostToDevice)); 


        // call the kernel
        striped_shared_memory_SPMV_kernel<<<num_blocks, num_threads, size_shared_memory>>>(
            rows_per_sm, num_x_elem, num_conseq_mem_reg,
            min_j_d, max_j_d,
            striped_A_d, num_rows, num_stripes, j_min_i,
            x_d, y_d
        );

        // synchronize the device
        CHECK_CUDA(hipGetLastError());
        CHECK_CUDA(hipDeviceSynchronize());
    }

// explicit template instantiation
template class Striped_Shared_Memory_Implementation<DataType>;