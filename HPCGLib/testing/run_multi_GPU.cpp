#include <testing.hpp>
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_mpi_utils.cuh"

#include <mpi.h>
#include <cuda_runtime.h>
#include <typeinfo>

#include <time.h>

using DataType = double;

//number of processes in x, y, z
#define NPX 3
#define NPY 3
#define NPZ 3
//each process gets assigned problem size of NX x NY x NZ
#define NX 18
#define NY 18
#define NZ 18

#define MPIDataType MPI_DOUBLE

void test_matrix_distribution(int num_stripes_local, int num_stripes_global, int num_rows_local, int num_rows_global, DataType *striped_A_local_h, DataType *striped_A_global_h, Problem *problem){
    //verfify the partial matrix
    assert(num_stripes_global == num_stripes_local);
    assert(num_rows_global - NX*NPX*NY*NPY*NZ*NPZ == num_rows_local - NX*NY*NZ);
    if(VerifyPartialMatrix(striped_A_local_h, striped_A_global_h, num_stripes_local, problem)){
        if(problem->rank == 0){
            printf("++++++\n");
            printf("Partial matrix A was correctly generated on all processes\n");
            printf("++++++\n");
        }
    }else{
        printf("++++++Rank=%d: Partial matrix A is  NOT correct++++++\n", problem->rank);
    }
}

void test_SPMV(striped_multi_GPU_Implementation<DataType> implementation_multi_GPU, striped_Matrix<DataType>* A_local_striped, striped_Matrix<DataType>* A_global_striped, Halo *halo_p_d, Halo *halo_Ap_d, Problem *problem, int *j_min_i_d){
    //make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_p_d, problem);
    SetHaloZeroGPU(halo_Ap_d);

    clock_t start_halo_exchange, end_halo_exchange;
    start_halo_exchange = clock();
    ExchangeHalo(halo_p_d, problem);
    end_halo_exchange = clock();
    double time_halo_exchange = ((double) (end_halo_exchange - start_halo_exchange)) / CLOCKS_PER_SEC;

    //run SPMV on multi GPU
    clock_t start_multi_GPU, end_multi_GPU;
    start_multi_GPU = clock();
    implementation_multi_GPU.compute_SPMV(*A_local_striped, halo_p_d, halo_Ap_d, problem, j_min_i_d); //1st * 2nd = 3rd argument
    end_multi_GPU = clock();
    double time_multi_GPU = ((double) (end_multi_GPU - start_multi_GPU)) / CLOCKS_PER_SEC;
    MPI_Barrier(MPI_COMM_WORLD);
    //if(problem->rank == 0)printf("Rank=%d:\t SPMV Result for multiGPU computed.\n", problem->rank);
    MPI_Barrier(MPI_COMM_WORLD);

    //verify the multi GPU result
    if(problem->rank == 0){
        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = 0;
        }
        GatherResult(halo_Ap_d, problem, result_multi_GPU_h);
        MPI_Barrier(MPI_COMM_WORLD);
        //if(problem->rank == 0) printf("Rank=%d:\t SPMV Result gathered.\n", problem->rank);
        MPI_Barrier(MPI_COMM_WORLD);
        
        //compute verification result on single GPU
        DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_single_GPU_h[i] = 0;
        }
        
        //create p_global
        DataType *p_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            p_global_h[i] = i;
        }
        DataType *p_global_d;
        CHECK_CUDA(cudaMalloc(&p_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(p_global_d, p_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //create Ap_global
        DataType *Ap_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            Ap_global_h[i] = 0;
        }
        DataType *Ap_global_d;
        CHECK_CUDA(cudaMalloc(&Ap_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(Ap_global_d, Ap_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //run SPMV on single GPU
        striped_warp_reduction_Implementation<DataType> implementation_single_GPU;
        //if(problem->rank == 0) printf("Rank=%d:\t SPMV About to do single GPU computation.\n", problem->rank);
        clock_t start_single_GPU, end_single_GPU;
        start_single_GPU = clock();
        implementation_single_GPU.compute_SPMV(*A_global_striped, p_global_d, Ap_global_d);
        end_single_GPU = clock();
        double time_single_GPU = ((double) (end_single_GPU - start_single_GPU)) / CLOCKS_PER_SEC;

        //copy result to CPU
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, Ap_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        //compare multiGPU result with single GPU result
        bool correct = true;
        double count = 0;
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            if(result_multi_GPU_h[i] != result_single_GPU_h[i]){
                if(count<10)
                    printf("Error: SPMV result_multi_GPU_h != result_single_GPU_h.\t index=%d,\t result_single_GPU_h[i]=%f,\t result_multi_GPU_h[i]%f\n", i, result_single_GPU_h[i], result_multi_GPU_h[i]);
                correct = false;
                count++;
            }
        }
        if(correct){
            printf("++++++\n");
            printf("SPMV is correct for multi GPU\n");
            printf("++++++\n");
            //printf("Time for multi GPU SPMV: %f\n", time_multi_GPU);
            //printf("Time for single GPU SPMV: %f\n", time_single_GPU);
            //printf("Halo Exchange Time: %f\n", time_halo_exchange);
        }else{
            printf("!!!!!SPMV Result is NOT correct!!!!!\n");
            double gn = NPX*NX*NPY*NY*NPZ*NZ;
            printf("SPMV: %f of %f wrong values which is %f percent\n", count, gn, count/(gn)*100.0);
        }
        free(result_single_GPU_h);
        free(result_multi_GPU_h);
        free(p_global_h);
        free(Ap_global_h);
        CHECK_CUDA(cudaFree(p_global_d));
        CHECK_CUDA(cudaFree(Ap_global_d));
    }else{
        SendResult(0, halo_Ap_d, problem);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

}

void test_SymGS(striped_multi_GPU_Implementation<DataType> implementation_multi_GPU, striped_Matrix<DataType>* A_local_striped, striped_Matrix<DataType>* A_global_striped, Halo *halo_p_d, Halo *halo_Ap_d, Problem *problem, int *j_min_i_d){
    //make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_p_d, problem);
    SetHaloGlobalIndexGPU(halo_Ap_d, problem);

    //exchange halo
    ExchangeHalo(halo_p_d, problem);
    ExchangeHalo(halo_Ap_d, problem);

    //run SymGS on multi GPU
    clock_t start_multi_GPU, end_multi_GPU;
    start_multi_GPU = clock();
    implementation_multi_GPU.compute_SymGS(*A_local_striped, halo_p_d, halo_Ap_d, problem, j_min_i_d); //1st * 2nd = 3rd argument
    end_multi_GPU = clock();
    double time_multi_GPU = ((double) (end_multi_GPU - start_multi_GPU)) / CLOCKS_PER_SEC;
    MPI_Barrier(MPI_COMM_WORLD);
    //if(problem->rank == 0) printf("Rank=%d:\t SymGS Result for multiGPU computed.\n", problem->rank);
    MPI_Barrier(MPI_COMM_WORLD);

    //verify the multi GPU result
    if(problem->rank == 0){
        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = i;
        }
        GatherResult(halo_p_d, problem, result_multi_GPU_h);
        MPI_Barrier(MPI_COMM_WORLD);
        //if(problem->rank == 0) printf("Rank=%d:\t SymGS Result gathered.\n", problem->rank);
        MPI_Barrier(MPI_COMM_WORLD);
        
        //compute verification result on single GPU
        
        //create p_global
        DataType *p_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            p_global_h[i] = i;
        }
        DataType *p_global_d;
        CHECK_CUDA(cudaMalloc(&p_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(p_global_d, p_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));
        
        //create Ap_global
        DataType *Ap_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            Ap_global_h[i] = i;
        }
        DataType *Ap_global_d;
        CHECK_CUDA(cudaMalloc(&Ap_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(Ap_global_d, Ap_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));
        
        //run SymGS on single GPU
        striped_box_coloring_Implementation<DataType> implementation_single_GPU;
        //if(problem->rank == 0) printf("Rank=%d:\t SPMV About to do single GPU computation.\n", problem->rank);
        clock_t start_single_GPU, end_single_GPU;
        start_single_GPU = clock();
        implementation_single_GPU.compute_SymGS(*A_global_striped, p_global_d, Ap_global_d);
        end_single_GPU = clock();
        double time_single_GPU = ((double) (end_single_GPU - start_single_GPU)) / CLOCKS_PER_SEC;
        
        //copy result to CPU
        DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_single_GPU_h[i] = 0;
        }
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, p_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        //compare multiGPU result with single GPU result
        bool correct = true;
        double count = 0;
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            if(result_multi_GPU_h[i] != result_single_GPU_h[i]){
                if(count<100)
                    printf("Error: SymGS result_multi_GPU_h != result_single_GPU_h.\t index=%d,\t result_single_GPU_h[i]=%f,\t result_multi_GPU_h[i]%f\n", i, result_single_GPU_h[i], result_multi_GPU_h[i]);
                correct = false;
                count++;
            }
        }
        if(correct){
            printf("++++++\n");
            printf("SymGS is correct for multi GPU\n");
            printf("++++++\n");
            //printf("Time for multi GPU SymGS: %f\n", time_multi_GPU);
            //printf("Time for single GPU SymGS: %f\n", time_single_GPU);
        }else{
            printf("!!!!!SymGS Result is NOT correct!!!!!\n");
            double gn = NPX*NX*NPY*NY*NPZ*NZ;
            printf("SymGS: %f of %f wrong values which is %f percent\n", count, gn, count/(gn)*100.0);
        }
        free(result_single_GPU_h);
        free(result_multi_GPU_h);
        free(p_global_h);
        free(Ap_global_h);
        CHECK_CUDA(cudaFree(p_global_d));
        CHECK_CUDA(cudaFree(Ap_global_d));
    }else{
        SendResult(0, halo_p_d, problem);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

}

void test_WAXPBY(striped_multi_GPU_Implementation<DataType> implementation_multi_GPU, striped_Matrix<DataType>* A_global_striped, Halo *halo_w_local_d, Halo *halo_x_local_d, Halo *halo_y_local_d, Problem *problem){
    bool all_passed = true;
    for(DataType alpha = -1.0; alpha <= 1.0; alpha += 0.5){
        for(DataType beta = -1.0; beta<= 1.0; beta += 0.5){

            //make sure that we work on clean data
            SetHaloGlobalIndexGPU(halo_w_local_d, problem);
            SetHaloGlobalIndexGPU(halo_x_local_d, problem);
            SetHaloGlobalIndexGPU(halo_y_local_d, problem);

            //run WAXPBY on multi GPU
            implementation_multi_GPU.compute_WAXPBY(halo_w_local_d, halo_y_local_d, halo_w_local_d, alpha, beta, problem, false); //1st * 2nd = 3rd argument
            MPI_Barrier(MPI_COMM_WORLD);
            //if(problem->rank == 0) printf("Rank=%d:\t WAXPBY Result for multiGPU computed.\n", problem->rank);
            MPI_Barrier(MPI_COMM_WORLD);

            //verify that the function did not write into the halo parts
            if(!IsHaloZero(halo_w_local_d)){
                printf("Error: alpha=%f, beta=%f wrote into halo\n", alpha, beta);
                all_passed = false;
            }

            //verify the multi GPU result
            if(problem->rank == 0){
                //gather result
                DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
                    result_multi_GPU_h[i] = i;
                }
                GatherResult(halo_w_local_d, problem, result_multi_GPU_h);
                MPI_Barrier(MPI_COMM_WORLD);
                //if(problem->rank == 0) printf("Rank=%d:\t WAXPBY Result gathered.\n", problem->rank);
                MPI_Barrier(MPI_COMM_WORLD);
                
                //compute verification result on single GPU
                
                //create p_global
                DataType *w_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
                    w_global_h[i] = i;
                }
                DataType *w_global_d;
                CHECK_CUDA(cudaMalloc(&w_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
                CHECK_CUDA(cudaMemcpy(w_global_d, w_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));
                
                DataType *x_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
                    x_global_h[i] = i;
                }
                DataType *x_global_d;
                CHECK_CUDA(cudaMalloc(&x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
                CHECK_CUDA(cudaMemcpy(x_global_d, x_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

                DataType *y_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
                    y_global_h[i] = i;
                }
                DataType *y_global_d;
                CHECK_CUDA(cudaMalloc(&y_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
                CHECK_CUDA(cudaMemcpy(y_global_d, y_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

                //run WAXPBY on single GPU
                striped_box_coloring_Implementation<DataType> implementation_single_GPU;
                //if(problem->rank == 0) printf("Rank=%d:\t WAXPBY About to do single GPU computation.\n", problem->rank);
                implementation_single_GPU.compute_WAXPBY(*A_global_striped, x_global_d, y_global_d, w_global_d, alpha, beta);
                
                //copy result to CPU
                DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
                    result_single_GPU_h[i] = 0;
                }
                CHECK_CUDA(cudaMemcpy(result_single_GPU_h, w_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
                
                //compare multiGPU result with single GPU result
                bool correct = true;
                double count = 0;
                for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
                    if(result_multi_GPU_h[i] != result_single_GPU_h[i]){
                        if(count == 0){
                            printf("Error: alpha=%f, beta=%f\n", alpha, beta);
                        }
                        if(count<10)
                            printf("Error: WAXPBY result_multi_GPU_h != result_single_GPU_h.\t index=%d,\t result_single_GPU_h[i]=%f,\t result_multi_GPU_h[i]%f\n", i, result_single_GPU_h[i], result_multi_GPU_h[i]);
                        all_passed = false;
                        correct = false;
                        count++;
                    }
                }

                free(result_single_GPU_h);
                free(result_multi_GPU_h);
                free(w_global_h);
                free(x_global_h);
                free(y_global_h);
                CHECK_CUDA(cudaFree(w_global_d));
                CHECK_CUDA(cudaFree(x_global_d));
                CHECK_CUDA(cudaFree(y_global_d));

            }else{
                SendResult(0, halo_w_local_d, problem);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    }
    if(all_passed){
        if(problem->rank == 0) {
            printf("++++++\n");
            printf("WAXPBY is correct for multi GPU\n");
            printf("++++++\n");
        }
    }else{
        if(problem->rank == 0) printf("++++++WAXPBY Result is NOT correct++++++\n");
    }
}

void test_Dot(striped_multi_GPU_Implementation<DataType> implementation_multi_GPU, striped_Matrix<DataType>* A_global_striped, Halo *halo_x_local_d, Halo *halo_y_local_d, Problem *problem){
    bool all_passed = true;

    //make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_x_local_d, problem);
    SetHaloGlobalIndexGPU(halo_y_local_d, problem);
    DataType *result_multi_GPU_d;
    CHECK_CUDA(cudaMalloc(&result_multi_GPU_d, sizeof(DataType)));
    CHECK_CUDA(cudaMemset(result_multi_GPU_d, 0, sizeof(DataType)));


    //run Dot on multi GPU
    implementation_multi_GPU.compute_Dot(halo_x_local_d, halo_y_local_d, result_multi_GPU_d); 
    MPI_Barrier(MPI_COMM_WORLD);
    //if(problem->rank == 0) printf("Rank=%d:\t Dot Result for multiGPU computed.\n", problem->rank);
    MPI_Barrier(MPI_COMM_WORLD);

    DataType result_multi_GPU_h;
    CHECK_CUDA(cudaMemcpy(&result_multi_GPU_h, result_multi_GPU_d, sizeof(DataType), cudaMemcpyDeviceToHost));

    //verify the multi GPU result
    if(problem->rank == 0){

        DataType *result_multi_GPU_d;
        CHECK_CUDA(cudaMalloc(&result_multi_GPU_d, sizeof(DataType)));
        CHECK_CUDA(cudaMemset(result_multi_GPU_d, 0, sizeof(DataType)));

        //compute verification result on single GPU
        
        //create p_global
        DataType *x_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            x_global_h[i] = i;
        }
        DataType *x_global_d;
        CHECK_CUDA(cudaMalloc(&x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(x_global_d, x_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        DataType *y_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            y_global_h[i] = i;
        }
        DataType *y_global_d;
        CHECK_CUDA(cudaMalloc(&y_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(y_global_d, y_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        DataType *result_single_GPU_d;
        CHECK_CUDA(cudaMalloc(&result_single_GPU_d, sizeof(DataType)));
        CHECK_CUDA(cudaMemset(result_single_GPU_d, 0, sizeof(DataType)));

        //run Dot on single GPU
        striped_box_coloring_Implementation<DataType> implementation_single_GPU;
        //if(problem->rank == 0) printf("Rank=%d:\t Dot About to do single GPU computation.\n", problem->rank);
        implementation_single_GPU.compute_Dot(*A_global_striped, x_global_d, y_global_d, result_single_GPU_d);
        
        //copy result to CPU
        DataType result_single_GPU_h;
        CHECK_CUDA(cudaMemcpy(&result_single_GPU_h, result_single_GPU_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        
        //receive result from each rank compare
        bool correct = true;
        int count = 0;
        for(int i = 0; i<problem->size; i++){
            if(i!=0){
                MPI_Recv(&result_multi_GPU_h, 1, MPIDataType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if(result_multi_GPU_h != result_single_GPU_h){
                if(count == 0){
                    printf("Error: Dot result_multi_GPU_h != result_single_GPU_h for rank=%d.\t result_single_GPU_h=%f,\t result_multi_GPU_h=%f\n", i, result_single_GPU_h, result_multi_GPU_h);
                }
                correct = false;
                count++;
            }
        }
        if(correct){
            printf("++++++\n");
            printf("DOT is correct for multi GPU\n");
            printf("++++++\n");
        }else{
            printf("!!!!!Dot Result is NOT correct!!!!!\n");
            printf("%d ranks have wrong values\n", count);
        }

        free(x_global_h);
        free(y_global_h);
        CHECK_CUDA(cudaFree(x_global_d));
        CHECK_CUDA(cudaFree(y_global_d));
        CHECK_CUDA(cudaFree(result_single_GPU_d));
        CHECK_CUDA(cudaFree(result_multi_GPU_d));

    }else{
        MPI_Send(&result_multi_GPU_h, 1, MPIDataType, 0, 0, MPI_COMM_WORLD);
    }
}


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
    striped_Matrix<double>* A_global_striped = A_global.get_Striped();
    int *j_min_i_d = (*A_global_striped).get_j_min_i_d(); //we need to pass this as an argument to the multi GPU function since the offsets for a partial matrix are different
    DataType *striped_A_global_d = (*A_global_striped).get_values_d();
    global_int_t num_rows_global = (*A_global_striped).get_num_rows();
    int num_stripes_global = (*A_global_striped).get_num_stripes();
    DataType *striped_A_global_h = (DataType*) malloc(num_rows_global*num_stripes_global*sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(striped_A_global_h, striped_A_global_d, num_rows_global*num_stripes_global*sizeof(DataType), cudaMemcpyDeviceToHost));
    
    //initialize p and Ap
    Halo halo_p_d;
    InitHaloMemGPU(&halo_p_d, NX, NY, NZ);
    SetHaloGlobalIndexGPU(&halo_p_d, &problem);

    Halo halo_Ap_d;
    InitHaloMemGPU(&halo_Ap_d, NX, NY, NZ);
    SetHaloZeroGPU(&halo_Ap_d);

    //initialize w, x and y
    Halo halo_w_d;
    InitHaloMemGPU(&halo_w_d, NX, NY, NZ);
    SetHaloZeroGPU(&halo_w_d);

    Halo halo_x_d;
    InitHaloMemGPU(&halo_x_d, NX, NY, NZ);
    SetHaloGlobalIndexGPU(&halo_x_d, &problem);

    Halo halo_y_d;
    InitHaloMemGPU(&halo_y_d, NX, NY, NZ);
    SetHaloGlobalIndexGPU(&halo_y_d, &problem);

    // create an instance of the version to run the functions on
    striped_multi_GPU_Implementation<DataType> implementation_multi_GPU;

    //test matrix distribution
    test_matrix_distribution(num_stripes_local, num_stripes_global, num_rows_local, num_rows_global, striped_A_local_h, striped_A_global_h, &problem);

    //test SPMV
    test_SPMV(implementation_multi_GPU, A_local_striped, A_global_striped, &halo_p_d, &halo_Ap_d, &problem, j_min_i_d);
    
    //test SymGS
    test_SymGS(implementation_multi_GPU, A_local_striped, A_global_striped, &halo_p_d, &halo_Ap_d, &problem, j_min_i_d);

    //test WAXPBY
    test_WAXPBY(implementation_multi_GPU, A_global_striped, &halo_w_d, &halo_x_d, &halo_y_d, &problem);

    //test Dot
    test_Dot(implementation_multi_GPU, A_global_striped, &halo_x_d, &halo_y_d, &problem);

    // free the memory
    FreeHaloGPU(&halo_Ap_d);
    FreeHaloGPU(&halo_p_d);
    FreeHaloGPU(&halo_w_d);
    FreeHaloGPU(&halo_x_d);
    FreeHaloGPU(&halo_y_d);
    free(striped_A_local_h);
    free(striped_A_global_h);


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("Rank=%d:\t Done.\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}