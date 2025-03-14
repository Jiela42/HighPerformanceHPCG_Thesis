#include "HPCG_versions/striped_multi_GPU.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.cuh"
#include "UtilLib/hpcg_mpi_utils.cuh"
// #include "MatrixLib/coloring.cuh"
// #include <iostream>
// #include <cuda_runtime.h>

__inline__ __device__ global_int_t local_i_to_global_i(
    int i, 
    int nx, int ny, int nz, 
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0
    )
    {
        int local_i_x = i % nx;
        int local_i_y = (i % (nx * ny)) / nx;
        int local_i_z = i / (nx * ny);
        return gi0 + local_i_x + local_i_y * gnx + local_i_z * (gnx * gny);
}

__inline__ __device__ local_int_t global_i_to_halo_i(
    int i,
    int nx, int ny, int nz,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
    )
    {
        local_int_t global_j_x = i % gnx;
        local_int_t global_j_y = (i % (gnx * gny)) / gnx;
        local_int_t global_j_z = i / (gnx * gny);
        int halo_j_x = global_j_x - px * nx + 1;
        int halo_j_y = global_j_y - py * ny + 1;
        int halo_j_z = global_j_z - pz * nz + 1;
        return halo_j_x + halo_j_y * (nx+2) + halo_j_z * ((nx+2) * (ny+2));
}

__global__ void striped_box_coloring_half_SymGS_kernel(
    int cooperation_number,
    int color, int bx, int by, int bz,
    int nx, int ny, int nz,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    DataType * x, DataType * y,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
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

    // How is the vector colored
    int color_offs_x = color % bx; //gives x-xcoordinate of first appearance of color
    int color_offs_y = (color - color_offs_x) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    int color_offs_z = (color - color_offs_x - bx * color_offs_y) / (bx * by); //gives z-coordinate of first appearance of color

    num_color_cols = (color_offs_x < nx % bx) ? (num_color_cols + 1) : num_color_cols;
    num_color_rows = (color_offs_y < ny % by) ? (num_color_rows + 1) : num_color_rows;
    num_color_faces = (color_offs_z < nz % bz) ? (num_color_faces + 1) : num_color_faces;

    int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

    for (int i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        
        // find out the position of the node (only considering faces, cols and rows that actually have that color)
        int ix = i % num_color_cols;
        int iy = ((i % (num_color_cols * num_color_rows))) / num_color_cols;
        int iz = i / (num_color_cols * num_color_rows);
        
        // adjust the counter to the correct position when all nodes are considered
        ix = ix * bx + color_offs_x;
        iy = iy * by + color_offs_y;
        iz = iz * bz + color_offs_z;

        //compute the local index of the node and convert to global index
        local_int_t li = ix + iy * nx + iz * nx * ny;

        global_int_t gi = local_i_to_global_i(li, nx, ny, nz, gnx, gny, gnz, gi0);
        DataType my_sum = 0.0;
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            global_int_t gj = j_min_i[stripe] + gi;
            if (gj>= 0 && gj < gnx * gny * gnz) {
                //convert gj to halo coordinate hj which is the memory location of gj in the halo struct
                local_int_t hj =  global_i_to_halo_i(gj, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
                my_sum -= striped_A[li * num_stripes + stripe] * x[hj];
            }
        }

        // reduce the my_sum using warp reduction
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            local_int_t hi =  global_i_to_halo_i(gi, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
            DataType diag = striped_A[li * num_stripes + diag_offset];
            DataType sum = diag * x[hi] + y[hi] + my_sum;
            x[hi] = sum / diag;           
        }
        __syncthreads();
    }
}

/*
__global__ void striped_box_coloring_half_SymGS_kernel(
    int cooperation_number,
    int color, int bx, int by, int bz,
    int nx, int ny, int nz,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    double * x, double * y,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int coop_group_id = tid / cooperation_number;
    int lane = tid % cooperation_number;
    int num_coop_groups = blockDim.x * gridDim.x / cooperation_number;

    // How is the vector colored
    int color_offs_x = color % bx; //gives x-xcoordinate of first appearance of color
    int color_offs_y = (color - color_offs_x) % (bx * by) / bx; //gives y-coordinate of first appearance of color
    int color_offs_z = (color - color_offs_x - bx * color_offs_y) / (bx * by); //gives z-coordinate of first appearance of color

    //first appearance of that color in local space, we need to adjust for the fact that nx, ny, nz might not be divisible by bx, by, bz and hence the first appearance of the color might not be at 0, 0, 0 locally
    global_int_t cx0 = (color_offs_x + bx - (nx % bx)) % bx;
    global_int_t cy0 = (color_offs_y + by - (ny % by)) % by;
    global_int_t cz0 = (color_offs_z + bz - (nz % bz)) % bz;

    //how often does this color appear in each direction
    int num_color_in_x = 1 + (nx - cx0) / bx;
    int num_color_in_y = 1 + (ny - cy0) / by;
    int num_color_in_z = 1 + (nz - cz0) / bz;
    int num_nodes_with_color = num_color_in_x * num_color_in_y * num_color_in_z;

    for (int i = coop_group_id; i < num_nodes_with_color; i += num_coop_groups){
        
        // find out the position of the node based on the number of nodes with this color in each direction
        int iz = i % num_color_in_x;
        int iy = (i % (num_color_in_x * num_color_in_y)) / num_color_in_x;
        int ix = i / (num_color_in_x * num_color_in_y);
        
        // adjust the counter to the correct position when all nodes are considered
        ix = cx0 + ix * bx;
        iy = cy0 + iy * by;
        iz = cz0 + iz * bz;
        //guard against out of bounds
        if(ix >= nx || iy >= ny || iz >= nz || ix < 0 || iy < 0 || iz < 0){
            continue;
        }

        //compute the local index of the node and convert to global index
        local_int_t li = ix * ny * nz + iy * nz + iz;
        global_int_t gi = local_i_to_global_i(li, nx, ny, nz, gnx, gny, gnz, gi0);
        DataType my_sum = 0.0;
        for(int stripe = lane; stripe < num_stripes; stripe += cooperation_number){
            global_int_t gj = j_min_i[stripe] + gi;
            if (gj >= 0 && gj < gnx * gny * gnz) {
                //convert gj to halo coordinate hj which is the memory location of gj in the halo struct
                local_int_t hj =  global_i_to_halo_i(gj, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
                my_sum -= striped_A[li * num_stripes + stripe] * x[hj];
            }
        }

        // reduce the my_sum using warp reduction
        for (int offset = cooperation_number/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }

        __syncthreads();
        if (lane == 0){
            local_int_t hi =  global_i_to_halo_i(gi, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
            double diag = striped_A[li * num_stripes + diag_offset];
            double sum = diag * x[hi] + y[hi] + my_sum;
            x[hi] = sum / diag;           
        }
        __syncthreads();
    }
}
*/

template <typename T>
void striped_multi_GPU_Implementation<T>::striped_box_coloring_multi_GPU_computeSymGS(
    striped_Matrix<T> & A,
    Halo *x_d, Halo *b_d, // the vectors x and y are already on the device
    Problem *problem,
    int *j_min_i
){
    //some geometrical information
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    assert(nx % 3 == 0);
    assert(ny % 3 == 0);
    assert(nz % 3 == 0);
    global_int_t gnx = problem->gnx;
    global_int_t gny = problem->gny;
    global_int_t gnz = problem->gnz;
    global_int_t gi0 = problem->gi0;
    int px = problem->px;
    int py = problem->py;
    int pz = problem->pz;
    
    int diag_offset = A.get_diag_index();

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int num_stripes = A.get_num_stripes();
    DataType * striped_A_d = A.get_values_d();

    assert(diag_offset >= 0);

    // check that the box size does not violate dependencies
    // we assume a 3d 27pt stencil
    int bx = this->bx;
    int by = this->by;
    int bz = this->bz;

    assert(bx >= 3);
    assert(by >= 3);
    assert(bz >= 3);

    // std::cout << "bx: " << bx << " by: " << by << " bz: " << bz << std::endl;
    /*
    int max_iterations = this->max_SymGS_iterations;
    double threshold_rr_Norm = 1.0;

    if(this->norm_based and max_iterations > 1){
        threshold_rr_Norm = this->getSymGS_rrNorm_zero_init(nx, ny, nz);
        assert(threshold_rr_Norm >= 0.0);
    }
    */

    int cooperation_number = this->SymGS_cooperation_number;

    // the number of blocks is now dependent on the maximum number of rows per color
    int num_colors = bx * by * bz;
    int max_color =  num_colors - 1;
    // std::cout << "max_color: " << max_color << std::endl;
    int max_num_rows_per_color = ceiling_division(nx, bx) * ceiling_division(ny, by) * ceiling_division(nz, bz);

    int num_blocks = std::min(ceiling_division(max_num_rows_per_color, 1024/cooperation_number), MAX_NUM_BLOCKS);
    /*
    double L2_norm_y;

    cudaStream_t y_Norm_stream;
    CHECK_CUDA(cudaStreamCreate(&y_Norm_stream));

    printf("ALERT: L2_norm_for_Device_Vector is not implemented for multi GPU!  Result is going to be wrong!");
    //L2_norm_for_Device_Vector(y_Norm_stream, num_rows, b_d, &L2_norm_y);
    */
    // to do the L2 norm asynchroneously we do the first iteration outside of the loop
    for(int color = 0; color < max_color; color++){
            // we need to do a forward pass
            striped_box_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
            cooperation_number,
            color, bx, by, bz,
            nx, ny, nz,
            num_rows, num_cols,
            num_stripes, diag_offset,
            j_min_i,
            striped_A_d,
            x_d->x_d, b_d->x_d,
            gnx, gny, gnz, gi0, px, py, pz
            );
            CHECK_CUDA(cudaDeviceSynchronize());
            ExchangeHalo(x_d, problem);
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
        x_d->x_d, b_d->x_d,
        gnx, gny, gnz, gi0, px, py, pz
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        ExchangeHalo(x_d, problem);
    }
    /*
    printf("ALERT: L2_norm_ is not implemented for multi GPU! Result is going to be wrong!");
    double L2_norm = 1.0; //L2_norm_for_SymGS(A, x_d, y_d);
    CHECK_CUDA(cudaStreamSynchronize(y_Norm_stream));
    CHECK_CUDA(cudaStreamDestroy(y_Norm_stream));

    double rr_norm = L2_norm / L2_norm_y;

    // std::cout << "rr_norm after one iteration: " << rr_norm << std::endl;

    int iter = 1;


    // this while loop only kicks in if we are benchmarking or testing SymGS itself
    // as a part of MG or CG we will not use this loop since in this case SymGS is only executed once
    while (iter < max_iterations and rr_norm > threshold_rr_Norm){

        for(int color = 0; color <= max_color; color++){
            // we need to do a forward pass
            striped_box_coloring_half_SymGS_kernel<<<num_blocks, 1024>>>(
            cooperation_number,
            color, bx, by, bz,
            nx, ny, nz,
            num_rows, num_cols,
            num_stripes, diag_offset,
            j_min_i,
            striped_A_d,
            x_d->x_d, b_d->x_d,
            gnx, gny, gnz, gi0, px, py, pz
            );
            CHECK_CUDA(cudaDeviceSynchronize());
            ExchangeHalo(x_d, problem);
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
            x_d->x_d, b_d->x_d,
            gnx, gny, gnz, gi0, px, py, pz
            );
            CHECK_CUDA(cudaDeviceSynchronize());
            ExchangeHalo(x_d, problem);
        }

        printf("ALERT: L2_norm_ is not implemented for multi GPU! Result is going to be wrong!");
        double L2_norm = 1.0; //L2_norm_for_SymGS(A, x_d, y_d);
   
        rr_norm = L2_norm / L2_norm_y;

        iter ++;
    }*/

    // std::cout << "SymGS for size " << nx << "x" << ny << "x" << nz << " took " << iter << " iterations." << std::endl;
    // std::cout << "RR norm after " << iter << " iterations: " << rr_norm << std::endl;
    // std::cout << "Threshold RR norm: " << threshold_rr_Norm << std::endl;
}

// explicit template instantiation
template class striped_multi_GPU_Implementation<DataType>;