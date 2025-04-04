#include <testing_hipified.hpp>
#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"
#include "HPCG_versions/striped_multi_GPU_hipified.cuh"
#include "MatrixLib/striped_partial_Matrix_hipified.hpp"
#include "HPCG_versions/blocking_mpi_halo_exchange_hipified.cuh"
#include "HPCG_versions/non_blocking_mpi_halo_exchange_hipified.cuh"

#include <mpi.h>
#include <hip/hip_runtime.h>

#include <time.h>

//number of processes in x, y, z
#define NPX 2
#define NPY 2
#define NPZ 1
//each process gets assigned problem size of NX x NY x NZ
#define NX 128
#define NY 128
#define NZ 128

/*
* MPI must be initializid before calling.
* Runs all tests for the multi GPU implementation on 3x3x3 processes with global problem size of 24x24x24.
* Multi GPU result is compared to single GPU result (warp striped version).
* Result is printed.
*/
void dimension_tests(int argc, char *argv[], striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU){


    Problem problem = *implementation_multi_GPU.init_comm(argc, argv, NPX, NPY, NPZ, NX, NY, NZ);
    //non_blocking_mpi_Implementation<DataType> implementation_multi_GPU_non_blocking_mpi;

    
    MPI_Barrier(MPI_COMM_WORLD);
    if(problem.rank == 0) printf("Testing started.\n");
    MPI_Barrier(MPI_COMM_WORLD);
    
    //set Device
    InitGPU(&problem);
    
    //initialize matrix partial matrix A_local
    /* striped_partial_Matrix<DataType> A_local(&problem);

    striped_partial_Matrix<DataType> *A_local_current = &A_local;
    for(int i = 0; i < 3; i++){
        A_local_current->initialize_coarse_Matrix();
        A_local_current = A_local_current->get_coarse_Matrix();
    } 
    
    //copy partial matrix to host to compare partital matrix with global matrix
    DataType *A_local_h = (DataType*) malloc(A_local.get_num_rows()*A_local.get_num_stripes()*sizeof(DataType));
    CHECK_CUDA(hipMemcpy(A_local_h, A_local.get_values_d(), A_local.get_num_rows()*A_local.get_num_stripes()*sizeof(DataType), hipMemcpyDeviceToHost));
    */

    //create global matrix for verification
    /*sparse_CSR_Matrix<DataType> A_global;
    A_global.generateMatrix_onGPU(NPX*NX, NPY*NY, NPZ*NZ);

    // create the coarse matrices for the mg routines
    sparse_CSR_Matrix <DataType>* current_matrix = &A_global;
    for(int i = 0; i < 3; i++){
        current_matrix->initialize_coarse_Matrix();
        current_matrix = current_matrix->get_coarse_Matrix();
    }

    striped_Matrix<DataType>* A_global_striped = A_global.get_Striped();
    
    //copy global matrix to host to compare partital matrix with global matrix
    DataType *A_global_h = (DataType*) malloc(A_global_striped->get_num_rows()*A_global_striped->get_num_stripes()*sizeof(DataType));
    CHECK_CUDA(hipMemcpy(A_global_h, A_global_striped->get_values_d(), A_global_striped->get_num_rows()*A_global_striped->get_num_stripes()*sizeof(DataType), hipMemcpyDeviceToHost));
    */

    //initialize p and Ap
    Halo halo_p_d;
    InitHalo(&halo_p_d, &problem);
    SetHaloGlobalIndexGPU(&halo_p_d, &problem);
    
    Halo halo_Ap_d;
    InitHalo(&halo_Ap_d, &problem);
    SetHaloZeroGPU(&halo_Ap_d);
    
    //initialize b, w, x and y
    Halo halo_w_d;
    InitHalo(&halo_w_d, &problem);
    SetHaloZeroGPU(&halo_w_d);
    
    Halo halo_x_d;
    InitHalo(&halo_x_d, &problem);
    SetHaloGlobalIndexGPU(&halo_x_d, &problem);
    
    Halo halo_y_d;
    InitHalo(&halo_y_d, &problem);
    SetHaloGlobalIndexGPU(&halo_y_d, &problem);
    
    Halo halo_b_d;
    InitHalo(&halo_b_d, &problem);
    SetHaloGlobalIndexGPU(&halo_b_d, &problem);

    //initialize halos with random data
    SetHaloRandomGPU(&halo_x_d, &problem, 0, 1, RANDOM_SEED);
    SetHaloRandomGPU(&halo_y_d, &problem, 0, 1, RANDOM_SEED);
    DataType *result_multi_GPU_d;
    CHECK_CUDA(hipMalloc(&result_multi_GPU_d, sizeof(DataType)));
    CHECK_CUDA(hipMemset(result_multi_GPU_d, 0, sizeof(DataType)));

    //run Dot on multi GPU
    implementation_multi_GPU.compute_Dot(&halo_x_d, &halo_y_d, result_multi_GPU_d); 

    striped_Matrix<DataType> A_striped;
    A_striped.Generate_striped_3D27P_Matrix_onGPU(NX, NY, NZ);

    // test partial matrix
    striped_partial_Matrix<DataType> A_part(&problem);
    DataType *x = (DataType*)malloc (sizeof(DataType) * A_part.get_num_rows()* 27);

    implementation_multi_GPU.compute_SPMV(A_part, &halo_p_d, &halo_Ap_d, &problem); //1st * 2nd = 3rd argument
    
    /* CHECK_CUDA(hipMemcpy(x, A_part.get_values_d(), sizeof(DataType) * num_rows_local * 27, hipMemcpyDeviceToHost));
    int flag =1;
    for (int i=0; i<NX; i++)
    for (int j=0; j<NY; j++)
    for (int k=0; k<NZ; k++) {
        int row_local = i + j* NX + k* NX*NY;
        int row_global = i+problem.gx0 + (j+problem.gy0)* problem.gnx + (k+problem.gz0)* problem.gnx*problem.gny;
        for (int l=0; l<27; l++)
        {
            int idx_local = row_local*27+l;
            int idx_global = row_global*27+l;
            if((x[idx_local] - striped_A_global_h[idx_global]) * (x[idx_local] - striped_A_global_h[idx_global]) > 1e-26)
            {
                flag=0;
                printf("Error at location (%d, %d), %e, %e\n", idx_local, l, x[idx_local], striped_A_global_h[idx_global]);
                break;
            }
        }
    }
    if (problem.rank == 0) {if(flag) printf("Partial matrix correctly generated.\n");}
 */
    A_part.initialize_coarse_matrix();
    DataType *x_c = (DataType*)malloc(sizeof(DataType) * A_part.get_num_rows()/8 * 27);
    CHECK_CUDA(hipMemcpy(x_c, A_part.get_coarse_Matrix()->get_values_d(), sizeof(DataType) * A_part.get_num_rows()/8 * 27, hipMemcpyDeviceToHost));
    /* flag=1;
    for (int i=0; i<NX/2; i++)
    for (int j=0; j<NY/2; j++)
    for (int k=0; k<NZ/2; k++) {
        int row_local = i + j* NX/2 + k* NX/2*NY/2;
        int row_global = i+problem.gx0/2 + (j+problem.gy0/2)* problem.gnx/2 + (k+problem.gz0/2)* problem.gnx/2*problem.gny/2;
        for (int l=0; l<27; l++)
        {
            int idx_local = row_local*27+l;
            int idx_global = row_global*27+l;
            if((x_c[idx_local] - striped_A_global_c_h[idx_global]) * (x_c[idx_local] - striped_A_global_c_h[idx_global]) > 1e-26)
            {
                flag=0;
                printf("Error at location (%d, %d), %e, %e\n", idx_local, l, x_c[idx_local], striped_A_global_c_h[idx_global]);
                break;
            }
        }
    }
    if (problem.rank == 0) {if(flag) printf("Partial coarse matrix correctly generated.\n");}
 */
    free(x);
    free(x_c);
 
    // single GPU A_single 
    /* striped_Matrix<DataType> A_single;
    A_single.Generate_striped_3D27P_Matrix_onGPU(NX*NPX, NY*NPY, NZ*NPZ);
    DataType *a = (DataType*)malloc (sizeof(DataType) * num_rows_global * 27);
    CHECK_CUDA(hipMemcpy(a, A_single.get_values_d(), sizeof(DataType) * num_rows_global * 27, hipMemcpyDeviceToHost));
    int flag_=1;
    for (int i=0; i<NX*NPX; i++)
    for (int j=0; j<NY*NPY; j++)
    for (int k=0; k<NZ*NPZ; k++) {
        int row_global = i + j* NX*NPX + k* NX*NPX*NY*NPY;
        for (int l=0; l<27; l++)
        {
            int idx_global = row_global*27+l;
            if((a[idx_global] - striped_A_global_h[idx_global]) * (a[idx_global] - striped_A_global_h[idx_global]) > 1e-26)
            {
                flag_=0;
                printf("Error at location (%d, %d), %e, %e\n", idx_global, l, a[idx_global], striped_A_global_h[idx_global]);
                break;
            }
        }
    }
    if (problem.rank == 0) {if(flag_) printf("Striped matrix correctly generated.\n");}
    A_single.initialize_coarse_matrix();

    DataType *a_c = (DataType*)malloc (sizeof(DataType) * num_rows_global * 27);
    CHECK_CUDA(hipMemcpy(a_c, A_single.get_values_d(), sizeof(DataType) * num_rows_global * 27, hipMemcpyDeviceToHost));
    flag_=1;
    for (int i=0; i<NX*NPX/2; i++)
    for (int j=0; j<NY*NPY/2; j++)
    for (int k=0; k<NZ*NPZ/2; k++) {
        int row_global = i + j* NX*NPX/2 + k* NX*NPX/2*NY*NPY/2;
        for (int l=0; l<27; l++)
        {
            int idx_global = row_global*27+l;
            if((a_c[idx_global] - striped_A_global_h[idx_global]) * (a_c[idx_global] - striped_A_global_h[idx_global]) > 1e-26)
            {
                flag_=0;
                printf("Error at location (%d, %d), %e, %e\n", idx_global, l, a_c[idx_global], striped_A_global_h[idx_global]);
                break;
            }
        }
    }
    if (problem.rank == 0) {if(flag_) printf("Striped coarse matrix correctly generated.\n");}
 */
    //uncomment to compare ExchangeHalo implementations
    /* if (problem.rank==0) {printf("BEFORE\n"); PrintHalo(&halo_p_d);}
    implementation_multi_GPU.ExchangeHalo(&halo_p_d, &problem);
    if (problem.rank==0) {printf("AFTER\n"); PrintHalo(&halo_p_d);} */
      
    
    /* int iters = 10;
    double time_halo_exchange_blocking = 0;
    for(int i = 0; i < iters; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        implementation_multi_GPU.ExchangeHalo(&halo_p_d, &problem);
        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        time_halo_exchange_blocking += (end - start);
    }
    time_halo_exchange_blocking /= iters;
    
    double time_halo_exchange_non_blocking = 0;
    for(int i = 0; i < iters; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        implementation_multi_GPU_non_blocking_mpi.ExchangeHalo(&halo_p_d, &problem);
        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        time_halo_exchange_non_blocking += (end - start);
    }
    time_halo_exchange_non_blocking /= iters;
    */
    
    if(problem.rank == 0) {
        //printf("Time for halo exchange blocking: %f\n", time_halo_exchange_blocking);
        //printf("Time for halo exchange non-blocking: %f\n", time_halo_exchange_non_blocking);
        //printf("Speedup: %f\n", time_halo_exchange_blocking/time_halo_exchange_non_blocking);
    }

    // free the memory
    FreeHalo(&halo_p_d);
    FreeHalo(&halo_Ap_d);
    FreeHalo(&halo_w_d);
    FreeHalo(&halo_x_d);
    FreeHalo(&halo_y_d);
    FreeHalo(&halo_b_d);
    //free(A_local_h);
    //free(A_global_h);


    MPI_Barrier(MPI_COMM_WORLD);
    if(problem.rank == 0) printf("Testing done.\n", problem.rank);
    MPI_Barrier(MPI_COMM_WORLD);
    

    implementation_multi_GPU.finalize_comm(&problem);

}
