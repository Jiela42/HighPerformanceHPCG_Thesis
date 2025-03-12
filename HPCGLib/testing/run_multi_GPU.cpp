#include <testing.hpp>
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_mpi_utils.cuh"

#include <mpi.h>
#include <cuda_runtime.h>
#include <typeinfo>

using DataType = double;

//number of processes in x, y, z
#define NPX 3
#define NPY 3
#define NPZ 3
//each process gets assigned problem size of NX x NY x NZ
#define NX 16
#define NY 16
#define NZ 16

int main(int argc, char *argv[]){
    // this is supposed to show you how to run aNY of the functions the HPCG Library provides
    // we use a striped verison in this example

    MPI_Init( &argc , &argv );
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("Rank=%d:\t Initialized.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
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
    
    //create global matrix for verification
    sparse_CSR_Matrix<DataType> A_global;
    A_global.generateMatrix_onGPU(NPX*NX, NPY*NY, NPZ*NZ);
    striped_Matrix<double>* A_global_striped_d = A_global.get_Striped();
    int *j_min_i_d = (*A_global_striped_d).get_j_min_i_d(); //we need to pass this as an argument to the multi GPU function since the offsets for a partial matrix are different
    DataType *striped_A_global_d = (*A_global_striped_d).get_values_d();
    global_int_t num_rows_global = (*A_global_striped_d).get_num_rows();
    int num_stripes_global = (*A_global_striped_d).get_num_stripes();
    DataType *striped_A_global_h = (DataType*) malloc(num_rows_global*num_stripes_global*sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(striped_A_global_h, striped_A_global_d, num_rows_global*num_stripes_global*sizeof(DataType), cudaMemcpyDeviceToHost));

    //verfify the partial matrix
    assert(num_stripes_global == num_stripes_local);
    assert(num_rows_global - NX*NPX*NY*NPY*NZ*NPZ == num_rows_local - NX*NY*NZ);
    if(VerifyPartialMatrix(striped_A_local_h, striped_A_global_h, num_stripes_local, &problem)){
        //printf("++++++Rank=%d: Partial matrix A is correct++++++\n", rank);
    }else{
        printf("++++++Rank=%d: Partial matrix A is  NOT correct++++++\n", rank);
    }
    
    //initialize p and Ap
    Halo halo_p_d;
    InitHaloMemGPU(&halo_p_d, NX, NY, NZ);
    SetHaloGlobalIndexGPU(&halo_p_d, &problem);

    Halo halo_Ap_d;
    InitHaloMemGPU(&halo_Ap_d, NX, NY, NZ);
    SetHaloZeroGPU(&halo_Ap_d);

    //do halo exchange such that halos of p correctely initialized
    ExchangeHalo(&halo_p_d, &problem);

    // create an instance of the version to run the functions on
    striped_warp_reduction_Implementation<DataType> implementation;

    //run SPMV on multi GPU
    implementation.compute_SPMV_multi_GPU(*A_local_striped, halo_p_d.x_d, halo_Ap_d.x_d, &problem, j_min_i_d); //1st * 2nd = 3rd argument
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
        printf("Rank=%d:\t Result for multiGPU computed.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    //verify the multi GPU result
    if(rank == 0){
        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = 0;
        }
        GatherResult(&halo_Ap_d, &problem, result_multi_GPU_h);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)
            printf("Rank=%d:\t Result gathered.\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
        
        //compute verification result on single GPU
        DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_single_GPU_h[i] = 0;
        }
        
        //create p_global
        DataType *p_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            p_global_h[i] = 1.0/(i+1.0);
        }
        DataType *p_global_d;
        CHECK_CUDA(cudaMalloc(&p_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(p_global_d, p_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //create Ap_global
        DataType *Ap_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            Ap_global_h[i] = 0;
        }
        DataType *Ap_verify_d;
        CHECK_CUDA(cudaMalloc(&Ap_verify_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(Ap_verify_d, Ap_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //run SPMV on single GPU
        if(rank == 0)
            printf("Rank=%d:\t About to do single GPU computation.\n", rank);
        implementation.compute_SPMV(*A_global_striped_d, p_global_d, Ap_verify_d);

        //copy result to CPU
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, Ap_verify_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        //compare multiGPU result with single GPU result
        bool correct = true;
        double count = 0;
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            if(result_multi_GPU_h[i] != result_single_GPU_h[i]){
                if(count<10)
                    printf("Error: result_multi_GPU_h != result_single_GPU_h.\t index=%d,\t result_single_GPU_h[i]=%f,\t result_multi_GPU_h[i]%f\n", i, result_single_GPU_h[i], result_multi_GPU_h[i]);
                correct = false;
                count++;
            }
        }
        if(correct){
            printf("++++++Result is correct++++++\n");
        }else{
            printf("++++++Result is NOT correct++++++\n");
            double gn = NPX*NX*NPY*NY*NPZ*NZ;
            printf("%f of %f wrong values which is %f percent\n", count, gn, count/(gn)*100.0);
        }
    }else{
        SendResult(0, &halo_Ap_d, &problem);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // free the memory
    FreeHaloGPU(&halo_Ap_d);
    FreeHaloGPU(&halo_p_d);
    
    

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("Rank=%d:\t Done.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}