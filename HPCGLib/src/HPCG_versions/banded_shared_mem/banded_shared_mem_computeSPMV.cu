#include "HPCG_versions/naiveBanded.cuh"
#include "UtilLib/utils.cuh"
#include <cuda_runtime.h>

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


        // shared_j_min_i = num_bands * size_of(int);
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