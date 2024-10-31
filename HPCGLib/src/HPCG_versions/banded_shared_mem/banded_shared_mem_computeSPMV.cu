#include "HPCG_versions/banded_shared_mem.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>

void eliminate_overlap(int* min_j, int* max_j, int num_thick_bands, int* num_x_elem, int* num_consecutive_memory_regions){

    int ctr_consecutive_memory_regions = 1;

    for (int i = 1; i < num_thick_bands; i++){
        if (max_j[ctr_consecutive_memory_regions] <= min_j[i]){
            max_j[ctr_consecutive_memory_regions] = max_j[i];
        }
        else{
            min_j[ctr_consecutive_memory_regions] = min_j[i];
            max_j[ctr_consecutive_memory_regions] = max_j[i];
            num_consecutive_memory_regions ++;
        }
    }

    for (int i = 0; i < ctr_consecutive_memory_regions; i++){
        num_x_elem += max_j[i] - min_j[i];
    }
    *num_consecutive_memory_regions = ctr_consecutive_memory_regions;
}

template <typename T>
void Banded_Shared_Memory_Implementation<T>::banded_shared_memory_computeSPMV(
        banded_Matrix<T>& A, //we only pass A for the metadata
        T * banded_A_d, // the data of matrix A is already on the device
        int num_rows, int num_cols, // these refer to the shape of the banded matrix
        int num_bands, // the number of bands in the banded matrix
        int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the banded matrix
        T * x_d, T * y_d // the vectors x and y are already on the device
    ) {

        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = NUM_CORES_PER_SM * 4;
        int num_blocks = std::min(NUM_PHYSICAL_CORES, ceiling_division(num_rows, num_threads));

        // dynamically calculate how many rows can be handled at the same time
        // i.e. how many doubles are required per row

        std::vector<int> j_min_i_host = A.get_j_min_i();

        int new_elem_per_row = 1;

        for (int i = 1; i < num_bands; i++){
            if (j_min_i_host[i-1] + 1 != j_min_i_host[i]){
                new_elem_per_row ++;
            }
        }    

        int shared_mem_bytes = 1024 * SHARED_MEM_SIZE;
        int shared_mem_doubles = shared_mem_bytes / sizeof(double);

        int rows_per_sm = (shared_mem_doubles -  2 * num_bands) / new_elem_per_row;

        // based on the guess of the number of rows per sm we calculate the exact numbers
        
        int min_j [num_bands];
        int max_j [num_bands];
        
        // in the kernel we will need to adjust this, by adding the row start
        min_j[0] = j_min_i[0];
        max_j[0] = j_min_i[0] + rows_per_sm;

        // this refers to how many independent bands we end up having
        int num_thick_bands = 1;
        for (int band = 1; band < num_bands; band++){
            
            if (j_min_i[band-1] + 1 == j_min_i[band]){
                max_j[num_thick_bands-1] += 1;
            }
            else{
                // again in the kernel this will need to be adjusted
                min_j[num_thick_bands] = j_min_i[band];
                max_j[num_thick_bands] = j_min_i[band] + rows_per_sm;
            }
        }

        int num_conseq_mem_reg = 0;
        int num_x_elem = 0;
        eliminate_overlap(min_j, max_j, num_thick_bands, &num_x_elem, &num_conseq_mem_reg);

        int size_shared_j_min_i = num_bands * sizeof(int);
        int size_x_offsets = 3*num_conseq_mem_reg * sizeof(int);
        int size_shared_x = num_x_elem * sizeof(double);
        int size_shared_memory = size_shared_j_min_i  + size_x_offsets + size_shared_x;
        
        // move the offsets to the device
        // I am not sure if this is actually faster,
        // or if it would make sense to have the first thread in a block do this computation
        int * min_j_d;
        int * max_j_d;

        cudaMalloc(&min_j_d, num_bands * sizeof(int));
        cudaMalloc(&max_j_d, num_bands * sizeof(int));

        cudaMemcpy(min_j_d, min_j, num_bands * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(max_j_d, max_j, num_bands * sizeof(int), cudaMemcpyHostToDevice); 

        // call the kernel
        banded_shared_memory_SPMV_kernel<<<num_blocks, num_threads, size_shared_memory>>>(
            rows_per_sm, size_shared_x, num_conseq_mem_reg,
            min_j_d, max_j_d,
            banded_A_d, num_rows, num_bands, j_min_i,
            x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();
    }

// explicit template instantiation
template class Banded_Shared_Memory_Implementation<double>;