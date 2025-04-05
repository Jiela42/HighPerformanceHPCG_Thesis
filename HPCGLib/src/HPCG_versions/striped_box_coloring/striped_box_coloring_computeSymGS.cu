#include "HPCG_versions/striped_box_coloring.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"

// #include "MatrixLib/coloring.cuh"
// #include <iostream>
// #include <cuda_runtime.h>

__global__ void striped_box_coloring_half_SymGS_kernel(
    int cooperation_number,
    local_int_t color, int bx, int by, int bz,
    int nx, int ny, int nz,
    local_int_t num_rows, local_int_t num_cols,
    int num_stripes, int diag_offset,
    local_int_t * j_min_i,
    DataType * striped_A,
    DataType * x, DataType * y
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int coop_group_id = tid / cooperation_number;
    int lane = tid % cooperation_number;
    int num_coop_groups = blockDim.x * gridDim.x / cooperation_number;

    // now we do some calculations concerning the colors
    // cols are in the x direction
    // rows are in the y direction
    // stripes are in the z direction

    int num_color_cols = nx / bx;
    int num_color_rows = ny / by;
    int num_color_faces = nz / bz;

    int color_offs_x = color % bx;
    int color_offs_y = (color - color_offs_x) % (bx * by) / bx;
    int color_offs_z = (color - color_offs_x - bx * color_offs_y) / (bx * by);

    num_color_cols = (color_offs_x < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    local_int_t num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

    for (local_int_t i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        
        // find out the position of the node (only considering faces, cols and rows that actually have that color)
        int ix = i % num_color_cols;
        int iy = (i % (num_color_cols * num_color_rows)) / num_color_cols;
        int iz = i / (num_color_cols * num_color_rows);
        
        // adjust the counter to the correct position when all nodes are considered
        ix = ix * bx + color_offs_x;
        iy = iy * by + color_offs_y;
        iz = iz * bz + color_offs_z;

        local_int_t row = ix + iy * nx + iz * nx * ny;

        DataType my_sum = 0.0;
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            local_int_t col = j_min_i[stripe] + row;
            DataType val = striped_A[row * num_stripes + stripe];
            if(col < num_cols && col >= 0){
                my_sum -= val * x[col];
            }
        }
        
        // reduce the my_sum using warp reduction
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
        
        __syncthreads();
        if (lane == 0){
            DataType diag = striped_A[row * num_stripes + diag_offset];
            DataType sum = diag * x[row] + y[row] + my_sum;
            x[row] = sum / diag;           
        }
        __syncthreads();
    }
}


template <typename T>
void striped_box_coloring_Implementation<T>::striped_box_coloring_computeSymGS(
    striped_Matrix<T> & A,
    T * x_d, T * y_d // the vectors x and y are already on the device
){
    int diag_offset = A.get_diag_index();

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    local_int_t * j_min_i = A.get_j_min_i_d();
    T * striped_A_d = A.get_values_d();

    assert(diag_offset >= 0);

    // check that the box size does not violate dependencies
    // we assume a 3d 27pt stencil
    int bx = this->bx;
    int by = this->by;
    int bz = this->bz;

    assert(bx >= 2);
    assert(by >= 2);
    assert(bz >= 2);

    // std::cout << "bx: " << bx << " by: " << by << " bz: " << bz << std::endl;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    int max_iterations = this->max_SymGS_iterations;
    // double threshold_rr_Norm = 1.0;

    // if(this->norm_based and max_iterations > 1){
    //     threshold_rr_Norm = this->getSymGS_rrNorm_zero_init(nx, ny, nz);
    //     assert(threshold_rr_Norm >= 0.0);
    // }
    

    int cooperation_number = this->SymGS_cooperation_number;

    // the number of blocks is now dependent on the maximum number of rows per color
    int num_colors = bx * by * bz;
    int max_color =  num_colors - 1;
    // std::cout << "max_color: " << max_color << std::endl;
    local_int_t max_num_rows_per_color = ceiling_division(nx, bx) * ceiling_division(ny, by) * ceiling_division(nz, bz);
    
    int num_blocks = std::min(ceiling_division(max_num_rows_per_color, 1024/cooperation_number), MAX_NUM_BLOCKS);

    // double rr_norm = 1.0;

    // double L2_norm_y;

    // cudaStream_t y_Norm_stream;
    // CHECK_CUDA(cudaStreamCreate(&y_Norm_stream));
    // if(max_iterations > 1){
    
        // L2_norm_for_Device_Vector(y_Norm_stream, num_rows, y_d, &L2_norm_y);
    // }

    // int max_iterations = this->max_SymGS_iterations;
    // std::cout << "max_iterations = " << max_iterations << std::endl;
    double norm0 = 1.0;
    double normi = norm0;

    if(max_iterations != 1){
        // compute the original L2 norm
        norm0 = this->L2_norm_for_SymGS(A, x_d, y_d);
    }

    // std::cout << "normi/norm0 = " << normi/norm0 << std::endl;
    // std::cout << this->SymGS_tolerance << std::endl;

    // int total_iterations = 0;

    

    for(int i = 0; i < max_iterations && normi/norm0 > this->SymGS_tolerance; i++){

    
    // to do the L2 norm asynchroneously we do the first iteration outside of the loop
        for(int color = 0; color < num_colors; color++){
                // we need to do a forward pass
                striped_box_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
                cooperation_number,
                color, bx, by, bz,
                nx, ny, nz,
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

                striped_box_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
                cooperation_number,
                color, bx, by, bz,
                nx, ny, nz,
                num_rows, num_cols,
                num_stripes, diag_offset,
                j_min_i,
                striped_A_d,
                x_d, y_d
                );
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        // total_iterations ++;
        if(max_iterations > 1){

            // double L2_norm = this->L2_norm_for_SymGS(A, x_d, y_d);
            // CHECK_CUDA(cudaStreamSynchronize(y_Norm_stream));
            
            // rr_norm = L2_norm / L2_norm_y;
            normi = this->L2_norm_for_SymGS(A, x_d, y_d);
        }
        // CHECK_CUDA(cudaStreamDestroy(y_Norm_stream));

    }
    // std::cout << "SymGS for size " << nx << "x" << ny << "x" << nz << " took " << total_iterations << " iterations." << std::endl;
    // std::cout << "rr_norm after one iteration: " << rr_norm << std::endl;

    // int iter = 1;


    // // this while loop only kicks in if we are benchmarking or testing SymGS itself
    // // as a part of MG or CG we will not use this loop since in this case SymGS is only executed once
    // while (iter < max_iterations and rr_norm > threshold_rr_Norm){

    //     for(int color = 0; color <= max_color; color++){
    //         // we need to do a forward pass
    //         striped_box_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
    //         cooperation_number,
    //         color, bx, by, bz,
    //         nx, ny, nz,
    //         num_rows, num_cols,
    //         num_stripes, diag_offset,
    //         j_min_i,
    //         striped_A_d,
    //         x_d, y_d
    //         );
    //         CHECK_CUDA(cudaDeviceSynchronize());
    //     }

    //     // we need to do a backward pass,
    //     // the colors for this are the same just in reverse order
        
    //     for(int color = max_color; color  >= 0; color--){

    //         striped_box_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
    //         cooperation_number,
    //         color, bx, by, bz,
    //         nx, ny, nz,
    //         num_rows, num_cols,
    //         num_stripes, diag_offset,
    //         j_min_i,
    //         striped_A_d,
    //         x_d, y_d
    //         );
    //         CHECK_CUDA(cudaDeviceSynchronize());
    //     }

    //     double L2_norm = this->L2_norm_for_SymGS(A, x_d, y_d);
   
    //     rr_norm = L2_norm / L2_norm_y;

    //     iter ++;
    // }

    // std::cout << "SymGS for size " << nx << "x" << ny << "x" << nz << " took " << iter << " iterations." << std::endl;
    // std::cout << "RR norm after " << iter << " iterations: " << rr_norm << std::endl;
    // std::cout << "Threshold RR norm: " << threshold_rr_Norm << std::endl;
}

// explicit template instantiation
template class striped_box_coloring_Implementation<DataType>;