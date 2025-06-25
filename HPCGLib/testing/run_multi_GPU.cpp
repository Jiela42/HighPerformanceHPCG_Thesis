#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions/nccl_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_host_only_mpi_halo_exchange.cuh"

#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>

#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

using namespace cute;

using DataType = double;

#define MPIDataType MPI_DOUBLE
//number of processes in x, y, z
#define NPX 1
#define NPY 1
#define NPZ 1
//each process gets assigned problem size of NX x NY x NZ
#define NX 4
#define NY 4
#define NZ 4

int main(int argc, char *argv[]){

    int nx = NX;
    int ny = NY;
    int nz = NZ;
    
    blocking_mpi_Implementation<DataType> implementation_multi_GPU;

    Problem problem = *implementation_multi_GPU.init_comm(argc, argv, NPX, NPY, NPZ, NX, NY, NZ, true);

    MPI_Barrier(MPI_COMM_WORLD);
    if(problem.rank == 0){
        printf("Testing started.\n");
        printf("Comm Type: %s\n", implementation_multi_GPU.comm_type.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int x = 6;
    int y = 6;
    int z = 6;
    int bx = 3;
    int by = 3;
    int bz = 3;

    int num_colors = bx * by * bz;
    int max_num_per_color = ((x + bx - 1) / bx) * ((y + by - 1) / by) * ((z + bz - 1) / bz);
    int stencil_size = 27;
    Layout layout_3d_color_wise = make_layout(
        make_shape(num_colors, make_shape(stencil_size, max_num_per_color)), //num_colors blocks, each containing max_num_per_color * stencil_size coefficients
        make_stride(max_num_per_color * stencil_size, make_stride(max_num_per_color, 1)) //to skip a full color you skip max_num_per_color * stencil_size coefficient
    );

    //print layout_3d_color_wise
    printf("3D Color-wise Layout: ");
    print(layout_3d_color_wise);
    printf("\n");
    print_layout(layout_3d_color_wise);
    printf("\n");

    implementation_multi_GPU.finalize_comm(&problem);

    return 0;
}