#include "HPCG_versions/naiveBanded.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>

void eliminate_overlap(int* min_j, int* max_j, int num_thick_bands){

    int num_consecutive_memory_regions = 1;

    for (int i = 1; i < num_thick_bands; i++){
        if (max_j[num_consecutive_memory_regions] <= min_j[i]){
            max_j[num_consecutive_memory_regions] = max_j[i];
        }
        else{
            min_j[num_consecutive_memory_regions] = min_j[i];
            max_j[num_consecutive_memory_regions] = max_j[i];
            num_consecutive_memory_regions ++;
        }
    }
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

        std::vector * j_min_i_host = A.get_j_min_i();

        int new_elem_per_row = 1;

        for (int i = 1; i < num_bands; i++){
            if (not j_min_i_host[i-1] + 1 == j_min_i_host[i]){
                new_elem_per_row ++;
            }
        }    

        int shared_mem_bytes = 1024 * SHARED_MEM_SIZE;
        int shared_mem_doubles = shared_mem_bytes / sizeof(double);

        int rows_per_sm = (shared_Mem_doubles -  2 * num_bands) / new_elem_per_row

        // based on the guess of the number of rows per sm we calculate the exact numbers
        
        int [num_bands] min_j;
        int [num_bands] max_j;
        int [num_bands] consecutive_j;
        
        // in the kernel we will need to adjust this, by adding the row start
        min_j[0] = j_min_i[0];
        max_j[0] = j_min_i[0] + rows_per_sm;
        consecutive_j[0] = rows_per_sm;

        // this refers to how many independent bands we end up having
        int num_thick_bands = 1;

        for (int band = 1; band < num_bands; band++){
            
            if (j_min_i[band-1] + 1 == j_min_i[band]){
                max_j[num_thick_bands-1] += 1;
                consecutive_j[num_thick_bands-1] += 1
            }
            else{
                // again in the kernel this will need to be adjusted
                min_j[num_thick_bands] = j_min_i[band];
                max_j[num_thick_bands] = j_min_i[band] + num_rows_per_sm;
                consecutive_j[num_thick_bands] = num_rows_per_sm;
                num_thick_bands ++;
            }
        }
        
        // you need to re-work this!
        // in the eliminate overlap!!!
        //just hand in a pointer to an int.
        int num_x_elem = 0;
        for(int i = 0; i < num_thick_bands; i++){
            num_x_elem += consecutive_j[i];
        }

        eliminate_overlap(min_j, max_j, num_thick_bands);

        int size_shared_j_min_i = num_bands * size_of(int);
        int size_shared_x = num_x_elem * size_of(double);
        int size_x_offsets = 
        int size_shared_memory = size_shared_j_min_i + size_shared_x;
        // shared_x = (rows_per_sm -1) * new_elem_per_row + num_bands;

        // call the kernel
        banded_shared_memory_SPMV_kernel<<<num_blocks, num_threads>>>(
            num_rows_per_sm, banded_A_d, num_rows, num_bands, j_min_i, x_d, y_d
        );

        // synchronize the device
        cudaDeviceSynchronize();
    }

// explicit template instantiation
template class naiveBanded_Implementation<double>;