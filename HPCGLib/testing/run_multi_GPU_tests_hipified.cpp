#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"
#include "HPCG_versions/striped_multi_GPU_hipified.cuh"
#include "HPCG_versions/blocking_mpi_halo_exchange_hipified.cuh"
#include "HPCG_versions/non_blocking_mpi_halo_exchange_hipified.cuh"
//#include "HPCG_versions/nccl_halo_exchange_hipified.cuh"
#include "HPCG_versions_tests/kernel_multi_GPU_tests_hipified.cpp"
#include "HPCG_versions_tests/dim_check_hipified.cpp"
#include "MatrixLib/striped_partial_Matrix_hipified.hpp"

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <time.h>

using DataType = double;

#define MPIDataType MPI_DOUBLE

int main(int argc, char *argv[]){

    //blocking_mpi_Implementation<DataType> implementation_multi_GPU_blocking_mpi;
    //run_multi_GPU_tests(argc, argv, implementation_multi_GPU_blocking_mpi);
    
    non_blocking_mpi_Implementation<DataType> implementation_multi_GPU_non_blocking_mpi;
    //run_multi_GPU_tests(argc, argv, implementation_multi_GPU_non_blocking_mpi);
    dimension_tests(argc, argv, implementation_multi_GPU_non_blocking_mpi);
    //NCCL_Implementation<DataType> implementation_multi_GPU_nccl;
    //run_multi_GPU_tests(argc, argv, implementation_multi_GPU_nccl);

    return 0;
}