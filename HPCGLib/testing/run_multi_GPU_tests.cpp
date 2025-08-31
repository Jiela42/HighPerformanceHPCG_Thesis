#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_host_only_mpi_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_cuda_aware_mpi_halo_exchange.cuh"
#include "HPCG_versions/nccl_halo_exchange.cuh"
#include "HPCG_versions_tests/kernel_multi_GPU_tests.cpp"
#include "MatrixLib/striped_partial_Matrix.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>

using DataType = double;

#define MPIDataType MPI_DOUBLE

/*
* Can be used to test any implementation. Give the implementation type as a command line argument.
* Tests on two nodes with 4 GPUs each and local problem size of 32x32x32.
* example:
* srun -n 8 ./run_multi_GPU_tests BLOCKING_MPI
*/
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <implementation_type>\n";
        std::cerr << "Options: BLOCKING_MPI, NON_BLOCKING_HOST_MPI, NON_BLOCKING_CUDA_MPI, NCCL\n";
        return 1;
    }

    printf("TEST TEST TEST TEST\n");
    printf("TEST TEST TEST TEST\n");
    printf("TEST TEST TEST TEST\n");
    printf("TEST TEST TEST TEST\n");
    printf("TEST TEST TEST TEST\n");
    printf("TEST TEST TEST TEST\n");

    std::string impl_type = argv[1];
    if (impl_type == "BLOCKING_MPI") {
        blocking_mpi_Implementation<DataType> impl;
        run_multi_GPU_tests(argc, argv, impl);
    } else if (impl_type == "NON_BLOCKING_HOST_MPI") {
        non_blocking_host_only_mpi_Implementation<DataType> impl;
        run_multi_GPU_tests(argc, argv, impl);
    } else if (impl_type == "NON_BLOCKING_CUDA_MPI") {
        non_blocking_cuda_aware_mpi_Implementation<DataType> impl;
        run_multi_GPU_tests(argc, argv, impl);
    } else if (impl_type == "NCCL") {
        NCCL_Implementation<DataType> impl;
        run_multi_GPU_tests(argc, argv, impl);
    } else {
        std::cerr << "Unknown implementation type: " << impl_type << "\n";
        return 1;
    }

    return 0;
}