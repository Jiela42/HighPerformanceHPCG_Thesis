#include "HPCG_versions/AMGX.hpp"

// #include "amgx_wrapper.h"
// #include <thrust/system/cpp/memory.h> // amgx uses thrust
#include <amgx_c.h>
// #include "global_thread_handle.h"
// #include "../../../../../AMGX/include/amgx_c.h"
// #include <thrust.h>
// #include "thrust/iterator/detail/host_system_tag.h"
// #include "thrust/detail/config.h"
// #include "memory.h"
// #include "thrust/system/cpp/memory.h"


//////////////////// Files with namespace issues ////////////////////

// #include <amg.h>
// #include <multiply.h>
// #include <matrix.h>
// #include "cutil.h"
// #include "amgx_c_common.h"
// #include "resources.h"
// #include "amgx_cublas.h"
// #include "thread_manager.h"
// #include <cusp/memory.h>
// #include "cusp/detail/host/conversion_utils.h"
// #include "cusp/detail/dispatch/convert.h"

////////////////////////////////////////////////////////////////////


// using namespace amgx;
// using namespace thrust;


template <typename T>
void amgx_Implementation<T>::amgx_computeSPMV(
    sparse_CSR_Matrix<T> & A, // we pass A for the metadata
    int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
    T * x_d, T * y_d // the vectors x and y are already on the device
)
{   
    // // Initialize AMGX
    AMGX_initialize();
    AMGX_initialize_plugins();

    // Create AMGX resources
    AMGX_resources_handle rsrc = nullptr;
    AMGX_config_handle config = nullptr;
    AMGX_matrix_handle matrix = nullptr;
    AMGX_vector_handle vec_x = nullptr;
    AMGX_vector_handle vec_y = nullptr;

    // Create configuration
    AMGX_config_create(&config, "default");

    // Create resources
    AMGX_resources_create_simple(&rsrc, config);

    // Create matrix and vectors
    AMGX_matrix_create(&matrix, rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&vec_x, rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&vec_y, rsrc, AMGX_mode_dDDI);

    // Allocate host memory for matrix data
    int num_rows = A.get_num_rows();
    int num_nonzeros = A.get_nnz();
    int *A_row_ptr_h = new int[num_rows + 1];
    int *A_col_idx_h = new int[num_nonzeros];
    T *A_values_h = new T[num_nonzeros];

    // Copy matrix data from device to host
    cudaMemcpy(A_row_ptr_h, A_row_ptr_d, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_col_idx_h, A_col_idx_d, num_nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_values_h, A_values_d, num_nonzeros * sizeof(T), cudaMemcpyDeviceToHost);

     // Upload matrix data to AMGX
    AMGX_matrix_upload_all(matrix, num_rows, num_nonzeros, 1, 1, A_row_ptr_h, A_col_idx_h, A_values_h, nullptr);

    // Upload vectors to AMGX
    AMGX_vector_upload(vec_x, num_rows, 1, x_d);
    AMGX_vector_upload(vec_y, num_rows, 1, y_d);

    // first we convert the sparse matrix to the AMGX format

    // we call the AMGX function
    AMGX_matrix_vector_multiply(matrix, vec_x, vec_y);
    // we convert the result back to the CSR format
    AMGX_vector_download(vec_y, y_d);

    // Clean up
    AMGX_vector_destroy(vec_x);
    AMGX_vector_destroy(vec_y);
    AMGX_matrix_destroy(matrix);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(config);
    AMGX_finalize();
}

// explicit template instantiation
template class amgx_Implementation<double>;