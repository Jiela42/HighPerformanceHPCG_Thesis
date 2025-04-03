#include "hip/hip_runtime.h"
#include "HPCG_versions/striped_multi_GPU_hipified.cuh"
#include "UtilLib/utils_hipified.cuh"
#include <hip/hip_runtime.h>

__inline__ __device__ global_int_t local_i_to_global_i(
    local_int_t i,
    int nx, int ny, int nz,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0
    )
    {
        /*
        int local_i_x = i % nx;
        int local_i_y = (i % (nx * ny)) / nx;
        int local_i_z = i / (nx * ny);
        return gi0 + local_i_x + local_i_y * gnx + local_i_z * (gnx * gny);*/
        return gi0 + (i % nx) + (((i / nx) % ny) * gnx) + ((i / (nx * ny)) * (gnx * gny)); //should be equivalent to the above
}

__inline__ __device__ local_int_t global_i_to_halo_i(
    global_int_t i,
    int nx, int ny, int nz,
    global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gi0,
    int px, int py, int pz
    )
    {
        /*
        local_int_t global_j_x = i % gnx;
        local_int_t global_j_y = (i % (gnx * gny)) / gnx;
        local_int_t global_j_z = i / (gnx * gny);
        int halo_j_x = global_j_x - px * nx + 1;
        int halo_j_y = global_j_y - py * ny + 1;
        int halo_j_z = global_j_z - pz * nz + 1;
        return halo_j_x + halo_j_y * (nx+2) + halo_j_z * ((nx+2) * (ny+2));*/
        return ((i % gnx) - px * nx + 1) +
            ((((i / gnx) % gny) - py * ny + 1) * (nx + 2)) +
            (((i / (gnx * gny)) - pz * nz + 1) * ((nx + 2) * (ny + 2))); //should be equivalent to the above
}

__global__ void striped_warp_reduction_multi_GPU_SPMV_kernel(
        DataType* striped_A,
        local_int_t num_rows, int num_stripes, local_int_t * j_min_i,
        double* x, double* y, int nx, int ny, int nz,
        global_int_t gnx, global_int_t gny, global_int_t gnz,
        global_int_t gi0,
        int px, int py, int pz
    )
{
    // printf("striped_warp_reduction_SPMV_kernel\n");
    int cooperation_number = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % cooperation_number;

    // every thread computes one or more rows of the matrix
    for (local_int_t i = tid/cooperation_number; i < num_rows; i += (blockDim.x * gridDim.x)/cooperation_number) {
        // compute the matrix-vector product for the ith row
        // convert i to global index
        global_int_t gi = local_i_to_global_i(i, nx, ny, nz, gnx, gny, gnz, gi0);
        double sum_i = 0;
        for (int stripe = lane; stripe < num_stripes; stripe += cooperation_number) {
            local_int_t gj = gi + j_min_i[stripe]; //use the global index gi to find global index gj
            if (gj >= 0 && gj < gnx * gny * gnz) {
                //convert gj to halo coordinate hj which is the memory location of gj in the halo struct
                local_int_t hj =  global_i_to_halo_i(gj, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
                local_int_t current_row = i * num_stripes;
                sum_i += striped_A[current_row + stripe] * x[hj];
            }
        }

        // now let's reduce the sum_i to a single value using warp-level reduction
        for(int offset = cooperation_number/2; offset > 0; offset /= 2){
            sum_i += __shfl_down_sync(static_cast<unsigned long long>(0xFFFFFFFFFFFFFFFF), sum_i, offset);
        }

        __syncthreads();

        if (lane == 0){
            //convert gi to halo coordinate hi which is the memory location of gi in the halo struct
            local_int_t hi =  global_i_to_halo_i(gi, nx, ny, nz, gnx, gny, gnz, gi0, px, py, pz);
            y[hi] = sum_i;
        }
    }
}

template <typename T>
void striped_multi_GPU_Implementation<T>::striped_warp_reduction_multi_GPU_computeSPMV(
        striped_partial_Matrix<T>& A,
        Halo *x_d, Halo *y_d, // the vectors x and y are already on the device
        Problem *problem
    ) {

        //std::cout << "Rank="<< problem->rank <<"\t striped_warp_reduction_computeSPMV" << std::endl;

        local_int_t num_rows = A.get_num_rows();
        int num_stripes = A.get_num_stripes();
        //int * j_min_i = A.get_j_min_i_d();
        T * striped_A_d = A.get_values_d();

        // since every thread is working on one or more rows we need to base the number of threads on that
        int num_threads = 1024;
        int num_blocks = (problem->nx * problem->ny * problem->nz + num_threads - 1) / num_threads;

        // call the kernel
        striped_warp_reduction_multi_GPU_SPMV_kernel<<<num_blocks, num_threads>>>(
            striped_A_d, num_rows, num_stripes, A.get_j_min_i_d(), x_d->x_d, y_d->x_d, problem->nx, problem->ny, problem->nz,
            problem->gnx, problem->gny, problem->gnz, problem->gi0, problem->px, problem->py, problem->pz
        );

        // synchronize the device
        hipDeviceSynchronize();

        // std::cerr << "Assertion failed in function: " << __PRETTY_FUNCTION__ << std::endl;
        // assert(false);
    }

// explicit template instantiation
template class striped_multi_GPU_Implementation<DataType>;