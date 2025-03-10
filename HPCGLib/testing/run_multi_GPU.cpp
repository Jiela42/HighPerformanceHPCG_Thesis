#include <testing.hpp>
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
    int nx = 9;
    int ny = 9;
    int nz = 9;

    MPI_Init( &argc , &argv );
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank=%d initialized.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

    //set Device
    InitGPU(&problem);
    
    //initialize matrix A
    //TODO: right now we are generating own matrix per process instead of only correct part of whole matrix
    striped_Matrix<DataType> A_striped;
    sparse_CSR_Matrix <DataType> A;
    
    //initialize x
    Halo halo_x_d;
    InitHaloMemGPU(&halo_x_d, nx, ny, nz);
    SetHaloZeroGPU(&halo_x_d);
    
    //do halo exchange
    ExchangeHalo(&halo_x_d, &problem);
    
    // free the memory
    FreeHaloGPU(&halo_x_d);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank=%d done.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}