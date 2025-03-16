#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions_tests/kernel_multi_GPU_tests.cpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>

using DataType = double;

#define MPIDataType MPI_DOUBLE

int main(int argc, char *argv[]){

    MPI_Init( &argc , &argv );
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    run_multi_GPU_tests();

    MPI_Finalize();

    return 0;
}