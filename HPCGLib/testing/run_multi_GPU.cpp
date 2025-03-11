#include <testing.hpp>
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_mpi_utils.cuh"

#include <mpi.h>

using DataType = double;

int main(int argc, char *argv[]){
    // this is supposed to show you how to run any of the functions the HPCG Library provides
    // we use a striped verison in this example

    //number of processes in x, y, z
    int npx = 2;
    int npy = 2;
    int npz = 1;

    //each process gets assigned problem size of nx x ny x nz
    int nx = 64;
    int ny = 64;
    int nz = 64;

    MPI_Init( &argc , &argv );
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank=%d:\t Initialized.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

    //set Device
    InitGPU(&problem);
    
    //initialize matrix A
    //TODO: right now we are generating own matrix per process instead of only correct part of whole matrix
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    // create the coarse matrices for the mg routines
    sparse_CSR_Matrix <DataType>* current_matrix = &A;
    for(int i = 0; i < 3; i++){
        current_matrix->initialize_coarse_Matrix();
        current_matrix = current_matrix->get_coarse_Matrix();
    }
    // get the striped matrix
    striped_Matrix<double>* A_striped = A.get_Striped();

    //initialize x
    Halo halo_p_d;
    InitHaloMemGPU(&halo_p_d, nx, ny, nz);
    SetHaloZeroGPU(&halo_p_d);

    Halo halo_Ap_d;
    InitHaloMemGPU(&halo_Ap_d, nx, ny, nz);
    SetHaloZeroGPU(&halo_Ap_d);

    // create an instance of the version to run the functions on
    striped_warp_reduction_multi_GPU_Implementation<DataType> implementation;

    implementation.compute_SPMV(*A_striped, halo_p_d.x_d, halo_Ap_d.x_d, &problem); //1st * 2nd = 3rd argument
    
    //do halo exchange
    ExchangeHalo(&halo_Ap_d, &problem);
    
    // free the memory
    FreeHaloGPU(&halo_Ap_d);
    FreeHaloGPU(&halo_p_d);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank=%d:\t Done.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}