#include "HPCG_versions/striped_coloringPrecomputed.cuh"
#include "HPCG_versions/striped_coloring.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include "MatrixLib/coloring.cuh"


// note: this function uses the same kernel as the implementation striped_coloring (hence the import)
// the difference is that this version does not need to compute the coloring, it is already precomputed
template <typename T>
void striped_coloringPrecomputed_Implementation<T>::striped_coloringPrecomputed_computeSymGS(
    striped_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device
){
    int diag_offset = A.get_diag_index();

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    local_int_t * j_min_i = A.get_j_min_i_d();
    T * striped_A_d = A.get_values_d();

    // the coloring was already computed, we can grab the pointers from the striped matrix object
    local_int_t * color_pointer_d = A.get_color_pointer_d();
    local_int_t * color_sorted_rows_d = A.get_color_sorted_rows_d();

    assert(diag_offset >= 0);
    // this assertion is here such that we don't benchmark the coloring computation
    // usually if these pointers are null we can just call the generate coloring function on the matrix
    assert(color_pointer_d != nullptr);
    assert(color_sorted_rows_d != nullptr);
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();

    // the number of blocks is now dependent on the maximum number of rows per color

    int max_num_rows_per_color = std::min(nx * ny / 4, std::min(nx * nz / 2, ny * nz));
    int max_color = (nx-1) + 2 * (ny-1) + 4 * (nz-1);

    int num_blocks = std::min(ceiling_division(max_num_rows_per_color, 1024/WARP_SIZE), MAX_NUM_BLOCKS);
    
    int max_iterations = this->max_SymGS_iterations;
    // std::cout << "max_iterations = " << max_iterations << std::endl;
    double norm0 = 1.0;
    double normi = norm0;

    if(max_iterations != 1){
        // compute the original L2 norm
        norm0 = this->L2_norm_for_SymGS(A, x_d, y_d);
    }

    for(int i = 0; i < max_iterations && normi/norm0 > this->SymGS_tolerance; i++){
        for(int color = 0; color <= max_color; color++){
            // we need to do a forward pass
            striped_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
            color, color_pointer_d, color_sorted_rows_d,
            num_rows, num_cols,
            num_stripes, diag_offset,
            j_min_i,
            striped_A_d,
            x_d, y_d
            );
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    
        // we need to do a backward pass,
        // the colors for this are the same just in reverse order
        
        for(int color = max_color; color  >= 0; color--){
    
            striped_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
            color, color_pointer_d, color_sorted_rows_d,
            num_rows, num_cols,
            num_stripes, diag_offset,
            j_min_i,
            striped_A_d,
            x_d, y_d
            );
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if(max_iterations != 1){
            normi = this->L2_norm_for_SymGS(A, x_d, y_d);
        }
    }
    
}

// explicit template instantiation
template class striped_coloringPrecomputed_Implementation<DataType>;