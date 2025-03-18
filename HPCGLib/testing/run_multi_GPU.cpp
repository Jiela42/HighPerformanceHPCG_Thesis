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

int main(int argc, char *argv[]){

    // Declare variables for problem dimensions
    int NPX, NPY, NPZ, NX, NY, NZ;
    bool provided = false;
    // Check if user provided dimensions via command-line arguments
    if(argc >= 7) {
         NPX = std::atoi(argv[1]);
         NPY = std::atoi(argv[2]);
         NPZ = std::atoi(argv[3]);
         NX  = std::atoi(argv[4]);
         NY  = std::atoi(argv[5]);
         NZ  = std::atoi(argv[6]);
        provided = true;
    } else {
         // Fallback default values
         NPX = 3;
         NPY = 3;
         NPZ = 3;
         NX  = 8;
         NY  = 8;
         NZ  = 8;
    }

    MPI_Init( &argc , &argv );
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) printf("Initialized.\n");
    if(rank == 0) {
        if(provided){
            ("Using NPX=%d, NPY=%d, NPZ=%d, NX=%d, NY=%d, NZ=%d\n", NPX, NPY, NPZ, NX, NY, NZ)
        } else {
            printf("Using default values of NPX=%d, NPY=%d, NPZ=%d, NX=%d, NY=%d, NZ=%d\n", NPX, NPY, NPZ, NX, NY, NZ);
        }
    
    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(NPX, NPY, NPZ, NX, NY, NZ, size, rank, &problem);

    //set Device
    InitGPU(&problem);

    //initialize matrix partial matrix A_local
    sparse_CSR_Matrix<DataType> A_local;
    A_local.generateMatrix_onGPU(NX, NY, NZ);

    // get the striped matrix
    striped_Matrix<DataType>* A_local_striped = A_local.get_Striped();

    //initialize local matrix
    DataType *striped_A_local_d = (*A_local_striped).get_values_d();
    local_int_t num_rows_local = (*A_local_striped).get_num_rows();
    int num_stripes_local = (*A_local_striped).get_num_stripes();
    DataType *striped_A_local_h = (DataType*) malloc(num_rows_local*num_stripes_local*sizeof(DataType));
    GenerateStripedPartialMatrix(&problem, striped_A_local_h);
    CHECK_CUDA(cudaMemcpy(striped_A_local_d, striped_A_local_h, num_rows_local*num_stripes_local*sizeof(DataType), cudaMemcpyHostToDevice));

    //intialize j_min_i_d
    int gnx = NPX*NX;
    int gny = NPY*NY;
    int gnz = NPZ*NZ;
    int j_min_i_h[27] = {-gnx * (gny + 1) - 1, -gnx * (gny + 1), -gnx * (gny + 1) + 1, -gnx * gny - 1, -gnx * gny, -gnx * gny + 1, -gnx * (gny-1)-1, -gnx * (gny-1), -gnx * (gny-1)+1,-gnx-1, -gnx,-gnx+1, -1, 0, 1, gnx-1, gnx, gnx + 1, gnx * (gny - 1) - 1 , gnx * (gny - 1), gnx * (gny - 1) + 1, gnx * gny - 1, gnx * gny, gnx * gny + 1, gnx * (gny + 1) - 1, gnx * (gny + 1), gnx * (gny + 1) + 1};
    
    int *j_min_i_d;
    CHECK_CUDA(cudaMalloc(&j_min_i_d, 27*sizeof(int)));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, &j_min_i_h[0], 27*sizeof(int), cudaMemcpyHostToDevice););

    //initialize x and b
    Halo halo_x_d;
    InitHalo(&halo_x_d, NX, NY, NZ);

    Halo halo_b_d;
    InitHalo(&halo_b_d, NX, NY, NZ);
    SetHaloGlobalIndexGPU(&halo_b_d, &problem);

    // create an instance of the version to run the functions on
    non_blocking_mpi_Implementation<DataType> implementation_multi_GPU_non_blocking_mpi;

    //exchange halos so each process starts with the correct data
    implementation_multi_GPU_non_blocking_mpi.ExchangeHalo(&halo_b_d, &problem);

    //run CG on multi GPU
    int n_iters_local;
    DataType normr_local;
    DataType normr0_local;
    MPI_Barrier(MPI_COMM_WORLD); // synchronize all processes
    double start_time = MPI_Wtime();

    implementation_multi_GPU_non_blocking_mpi.compute_CG(*A_local_striped, 
                                                    &halo_b_d, 
                                                    &halo_x_d, 
                                                    n_iters_local, 
                                                    normr_local, 
                                                    normr0_local, 
                                                    &problem, 
                                                    j_min_i_d);

    MPI_Barrier(MPI_COMM_WORLD); // ensure all processes complete the call
    double end_time = MPI_Wtime();

    double elapsed_time = end_time - start_time;
    if(problem.rank == 0) {
        printf("CG computation time: %f seconds\n", elapsed_time);
    }
    if(rank == 0) printf("CG done with n_iters_local=%d.\n", n_iters_local);

    if(rank == 0) printf("Done.\n");

    free(striped_A_local_h);
    FreeHaloGPU(&halo_x_d);
    FreeHaloGPU(&halo_b_d);

    MPI_Finalize();

    return 0;
}