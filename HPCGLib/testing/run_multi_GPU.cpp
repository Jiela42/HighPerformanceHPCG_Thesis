#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_mpi_halo_exchange.cuh"
// #include "HPCG_versions_tests/kernel_multi_GPU_tests.cpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>

using DataType = double;

#define MPIDataType MPI_DOUBLE
//number of processes in x, y, z
#define NPX 2
#define NPY 2
#define NPZ 2
//each process gets assigned problem size of NX x NY x NZ
#define NX 256
#define NY 256
#define NZ 256

int main(int argc, char *argv[]){

    // create an instance of the version to run the functions on
    non_blocking_mpi_Implementation<DataType> implementation_multi_GPU_non_blocking_mpi;

    Problem *problem = implementation_multi_GPU_non_blocking_mpi.init_comm(argc, argv, NPX, NPY, NPZ, NX, NY, NZ);

    InitGPU(problem);

    //initialize matrix partial matrix A_local
    striped_partial_Matrix<DataType> A(problem);
    
    //initialize p and Ap
    Halo p;
    InitHalo(&p, problem);
    SetHaloZeroGPU(&p);

    Halo Ap;
    InitHalo(&Ap, problem);
    SetHaloZeroGPU(&Ap);


    //run SPMV on multi GPU
    clock_t start_multi_GPU, end_multi_GPU;
    start_multi_GPU = clock();
    implementation_multi_GPU_non_blocking_mpi.ExchangeHalo(&p, problem); //1st * 2nd = 3rd argument
    end_multi_GPU = clock();
    double time_multi_GPU = ((double) (end_multi_GPU - start_multi_GPU)) / CLOCKS_PER_SEC;

    if(problem->rank == 0){
        printf("Rank=%d:\t SPMV Result for multiGPU computed.\n", problem->rank);
        printf("Time for SPMV on multi GPU: %f\n", time_multi_GPU);
    }

    implementation_multi_GPU_non_blocking_mpi.finalize_comm(problem);

    return 0;
}