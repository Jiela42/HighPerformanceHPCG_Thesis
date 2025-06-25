#include <testing.hpp>
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "MatrixLib/striped_partial_Matrix.hpp"
#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_host_only_mpi_halo_exchange.cuh"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
/**
 * @file kernel_multi_GPU_tests.cpp
 * @brief Contains validation routines comparing multi-GPU HPCG kernel outputs to single-GPU reference results.
 *
 * Each test ensures functional correctness of multi-GPU implementations like SPMV, SymGS, CG, and MG
 * by comparing their outputs to known-good single-GPU results on small domains.
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

#include <testing.hpp>
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"
#include "MatrixLib/striped_partial_Matrix.hpp"
#include "HPCG_versions/blocking_mpi_halo_exchange.cuh"
#include "HPCG_versions/non_blocking_host_only_mpi_halo_exchange.cuh"

#include <cusparse_v2.h>

// MPI process grid and local problem size configuration
#define NPX 1
#define NPY 1
#define NPZ 1
//each process gets assigned problem size of NX x NY x NZ
#define NX 256
#define NY 256
#define NZ 256


/**
 * @brief Verifies correct matrix distribution across processes by comparing local and global representations.
 */
void test_matrix_distribution(int num_stripes_local, int num_stripes_global, int num_rows_local, int num_rows_global, DataType *striped_A_local_h, DataType *striped_A_global_h, Problem *problem){
    // verify the partial matrix
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

/**
 * @brief Compares result of SPMV on multi-GPU to single-GPU baseline for correctness.
 */
void test_SPMV(striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU, striped_partial_Matrix<DataType>* A_local_striped, striped_Matrix<DataType>* A_global_striped, Halo *halo_p_d, Halo *halo_Ap_d, Problem *problem, bool print_timing){
    // Make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_p_d, problem);
    SetHaloZeroGPU(halo_Ap_d);

    implementation_multi_GPU.ExchangeHalo(halo_p_d, problem);

    // Run SPMV on multi-GPU setup
    for(int i = 0; i < (print_timing ? 10 : 1); i++){
        auto start = std::chrono::high_resolution_clock::now();
        implementation_multi_GPU.compute_SPMV(*A_local_striped, halo_p_d, halo_Ap_d, problem); //1st * 2nd = 3rd argument
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        if(print_timing){
            printf("CPU wall-clock time for multi-GPU test_SPMV: %f ms\n", elapsed.count());
        }
    }

    // Verify the multi-GPU result
    if(problem->rank == 0){
        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = 0;
        }
        GatherResult(halo_Ap_d, problem, result_multi_GPU_h);
        
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

        // Run SPMV on single-GPU baseline
        striped_warp_reduction_Implementation<DataType> implementation_single_GPU;
        for(int i = 0; i < (print_timing ? 10 : 1); i++){
            auto start = std::chrono::high_resolution_clock::now();
            implementation_single_GPU.compute_SPMV(*A_global_striped, p_global_d, Ap_global_d);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;
            if(print_timing){
                printf("CPU wall-clock time for single-GPU test_SPMV: %f ms\n", elapsed.count());
            }
        }

        //copy result to CPU
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, Ap_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        // Compare multi-GPU result to single-GPU baseline
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
    }
}

/**
 * @brief Compares result of SymGS on multi-GPU to single-GPU baseline for correctness.
 */
void test_SymGS(striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU, striped_partial_Matrix<DataType>* A_local_striped, striped_Matrix<DataType>* A_global_striped, Halo *halo_p_d, Halo *halo_Ap_d, Problem *problem, bool print_timing){
    // Make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_p_d, problem);
    SetHaloGlobalIndexGPU(halo_Ap_d, problem);

    // Exchange halo
    implementation_multi_GPU.ExchangeHalo(halo_p_d, problem);
    implementation_multi_GPU.ExchangeHalo(halo_Ap_d, problem);

    // Run SymGS on multi-GPU setup

    for(int i = 0; i < (print_timing ? 10 : 1); i++){
        auto start = std::chrono::high_resolution_clock::now();
        implementation_multi_GPU.compute_SymGS(*A_local_striped, halo_p_d, halo_Ap_d, problem); //1st * 2nd = 3rd argument
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        if(print_timing){
            printf("CPU wall-clock time for multi-GPU compute_SymGS: %f ms\n", elapsed.count());
        }
    }

    // Verify the multi-GPU result
    if(problem->rank == 0){
        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = i;
        }
        GatherResult(halo_p_d, problem, result_multi_GPU_h);
        
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
        
        // Run SymGS on single-GPU baseline
        striped_box_coloring_Implementation<DataType> implementation_single_GPU;
        for(int i = 0; i < (print_timing ? 10 : 1); i++){
            auto start = std::chrono::high_resolution_clock::now();
            implementation_single_GPU.compute_SymGS(*A_global_striped, p_global_d, Ap_global_d);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;
            if(print_timing){
                printf("CPU wall-clock time for single-GPU compute_SymGS: %f ms\n", elapsed.count());
            }
        }
        
        //copy result to CPU
        DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_single_GPU_h[i] = 0;
        }
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, p_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        // Compare multi-GPU result to single-GPU baseline
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
    }
}

/**
 * @brief Compares result of WAXPBY on multi-GPU to single-GPU baseline for correctness, for various alpha/beta.
 */
void test_WAXPBY(striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU, striped_Matrix<DataType>* A_global_striped, Halo *halo_w_local_d, Halo *halo_x_local_d, Halo *halo_y_local_d, Problem *problem, bool print_timing){
    bool all_passed = true;
    for(DataType alpha = -1.0; alpha <= 1.0; alpha += 0.5){
        for(DataType beta = -1.0; beta<= 1.0; beta += 0.5){

            // Make sure that we work on clean data
            SetHaloGlobalIndexGPU(halo_w_local_d, problem);
            SetHaloGlobalIndexGPU(halo_x_local_d, problem);
            SetHaloGlobalIndexGPU(halo_y_local_d, problem);

            // Run WAXPBY on multi-GPU setup
            
            for(int i = 0; i < (print_timing ? 10 : 1); i++){
                auto start = std::chrono::high_resolution_clock::now();
                implementation_multi_GPU.compute_WAXPBY(halo_w_local_d, halo_y_local_d, halo_w_local_d, alpha, beta, problem, false); //1st * 2nd = 3rd argument
                auto stop = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = stop - start;
                if(print_timing){
                    printf("CPU wall-clock time for multi-GPU compute_WAXPBY: %f ms\n", elapsed.count());
                }
            }

            // Verify that the function did not write into the halo parts
            if(!IsHaloZero(halo_w_local_d)){
                printf("Error: alpha=%f, beta=%f wrote into halo\n", alpha, beta);
                all_passed = false;
            }

            // Verify the multi-GPU result
            if(problem->rank == 0){
                //gather result
                DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
                    result_multi_GPU_h[i] = i;
                }
                GatherResult(halo_w_local_d, problem, result_multi_GPU_h);
                
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

                // Run WAXPBY on single-GPU baseline
                striped_box_coloring_Implementation<DataType> implementation_single_GPU;
                for(int i = 0; i < (print_timing ? 10 : 1); i++){
                    auto start = std::chrono::high_resolution_clock::now();
                    implementation_single_GPU.compute_WAXPBY(*A_global_striped, x_global_d, y_global_d, w_global_d, alpha, beta);
                    auto stop = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> elapsed = stop - start;
                    if(print_timing){
                        printf("CPU wall-clock time for single-GPU compute_WAXPBY: %f ms\n", elapsed.count());
                    }
                }
                
                //copy result to CPU
                DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
                for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
                    result_single_GPU_h[i] = 0;
                }
                CHECK_CUDA(cudaMemcpy(result_single_GPU_h, w_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
                
                // Compare multi-GPU result to single-GPU baseline
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

/**
 * @brief Compares result of Dot product on multi-GPU to single-GPU baseline for correctness.
 */
void test_Dot(striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU, striped_Matrix<DataType>* A_global_striped, Halo *halo_x_local_d, Halo *halo_y_local_d, Problem *problem, bool print_timing){
    bool all_passed = true;

    // Initialize halos with random data
    SetHaloRandomGPU(halo_x_local_d, problem, 0, 1, RANDOM_SEED);
    SetHaloRandomGPU(halo_y_local_d, problem, 0, 1, RANDOM_SEED);
    DataType *result_multi_GPU_d;
    CHECK_CUDA(cudaMalloc(&result_multi_GPU_d, sizeof(DataType)));
    CHECK_CUDA(cudaMemset(result_multi_GPU_d, 0, sizeof(DataType)));

    // Run Dot on multi-GPU setup
    for(int i = 0; i < (print_timing ? 10 : 1); i++){
        auto start = std::chrono::high_resolution_clock::now();
        implementation_multi_GPU.compute_Dot(halo_x_local_d, halo_y_local_d, result_multi_GPU_d); 
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        if(print_timing){
            printf("CPU wall-clock time for multi-GPU compute_Dot: %f ms\n", elapsed.count());
        }
    }

    DataType result_multi_GPU_h;
    CHECK_CUDA(cudaMemcpy(&result_multi_GPU_h, result_multi_GPU_d, sizeof(DataType), cudaMemcpyDeviceToHost));

    // Verify the multi-GPU result
    if(problem->rank == 0){

        DataType *result_multi_GPU_d;
        CHECK_CUDA(cudaMalloc(&result_multi_GPU_d, sizeof(DataType)));
        CHECK_CUDA(cudaMemset(result_multi_GPU_d, 0, sizeof(DataType)));

        //compute verification result on single GPU
        
        //create p_global
        DataType *x_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        GatherResult(halo_x_local_d, problem, x_global_h);
        DataType *x_global_d;
        CHECK_CUDA(cudaMalloc(&x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(x_global_d, x_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        DataType *y_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        GatherResult(halo_y_local_d, problem, y_global_h);
        DataType *y_global_d;
        CHECK_CUDA(cudaMalloc(&y_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(y_global_d, y_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        DataType *result_single_GPU_d;
        CHECK_CUDA(cudaMalloc(&result_single_GPU_d, sizeof(DataType)));
        CHECK_CUDA(cudaMemset(result_single_GPU_d, 0, sizeof(DataType)));

        // Run Dot on single-GPU baseline
        striped_box_coloring_Implementation<DataType> implementation_single_GPU;
        for(int i = 0; i < (print_timing ? 10 : 1); i++){
            auto start = std::chrono::high_resolution_clock::now();
            implementation_single_GPU.compute_Dot(*A_global_striped, x_global_d, y_global_d, result_single_GPU_d);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;
            if(print_timing){
                printf("CPU wall-clock time for single-GPU compute_Dot: %f ms\n", elapsed.count());
            }
        }
        
        //copy result to CPU
        DataType result_single_GPU_h;
        CHECK_CUDA(cudaMemcpy(&result_single_GPU_h, result_single_GPU_d, sizeof(DataType), cudaMemcpyDeviceToHost));
        
        // Receive result from each rank and compare
        bool correct = true;
        int count = 0;
        for(int i = 0; i<problem->size; i++){
            if(i!=0){
                MPI_Recv(&result_multi_GPU_h, 1, MPIDataType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if(std::abs(result_multi_GPU_h - result_single_GPU_h) > 1e-10){
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
            printf("Difference: %20ef\n", result_single_GPU_h - result_multi_GPU_h);
        }

        free(x_global_h);
        free(y_global_h);
        CHECK_CUDA(cudaFree(x_global_d));
        CHECK_CUDA(cudaFree(y_global_d));
        CHECK_CUDA(cudaFree(result_single_GPU_d));
        CHECK_CUDA(cudaFree(result_multi_GPU_d));

    }else{
        SendResult(0, halo_x_local_d, problem);
        SendResult(0, halo_y_local_d, problem);
        MPI_Send(&result_multi_GPU_h, 1, MPIDataType, 0, 0, MPI_COMM_WORLD);
    }
}

/**
 * @brief Compares result of two CG iterations on multi-GPU to single-GPU baseline for correctness.
 */
void test_CG(striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU, striped_partial_Matrix<DataType>* A_local_striped, striped_Matrix<DataType>* A_global_striped, Halo *halo_b_local_d, Halo *halo_x_local_d, Problem *problem, bool print_timing){
    // Make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_b_local_d, problem);
    SetHaloZeroGPU(halo_x_local_d);

    // Exchange halos so each process starts with the correct data
    implementation_multi_GPU.ExchangeHalo(halo_b_local_d, problem);
    implementation_multi_GPU.ExchangeHalo(halo_x_local_d, problem);

    // Run CG on multi-GPU setup
    int n_iters_local;
    DataType normr_local;
    DataType normr0_local;
    implementation_multi_GPU.max_CG_iterations = 1;
    implementation_multi_GPU.doPreconditioning = false; //no preconditioning for this test

    for(int i = 0; i < (print_timing ? 10 : 1); i++){
        auto start = std::chrono::high_resolution_clock::now();
        implementation_multi_GPU.compute_CG(*A_local_striped, halo_b_local_d, halo_x_local_d, n_iters_local, normr_local, normr0_local, problem); //1st * 2nd = 3rd argument
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        if(print_timing){
            printf("CPU wall-clock time for multi-GPU compute_CG: %f ms\n", elapsed.count());
        }
    }
    
    // Verify the multi-GPU result
    if(problem->rank == 0){

        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = 0;
        }
        GatherResult(halo_x_local_d, problem, result_multi_GPU_h);
        
        //compute verification result on single GPU
        DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_single_GPU_h[i] = 0;
        }
        
        //create b_global_h
        DataType *b_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            b_global_h[i] = i;
        }
        DataType *b_global_d;
        CHECK_CUDA(cudaMalloc(&b_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(b_global_d, b_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //create x_global_h
        DataType *x_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            x_global_h[i] = 0;
        }
        DataType *x_global_d;
        CHECK_CUDA(cudaMalloc(&x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(x_global_d, x_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //declare variables for CG
        int n_iters_global;
        DataType normr_global;
        DataType normr0_global;

        // Run CG on single-GPU baseline
        striped_box_coloring_Implementation<DataType> implementation_single_GPU;
        implementation_single_GPU.doPreconditioning = false; //no preconditioning for this test
        implementation_single_GPU.max_CG_iterations = 1;
        for(int i = 0; i < (print_timing ? 10 : 1); i++){
            auto start = std::chrono::high_resolution_clock::now();
            implementation_single_GPU.compute_CG(*A_global_striped, b_global_d, x_global_d, n_iters_global, normr_global, normr0_global);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;
            if(print_timing){
                printf("CPU wall-clock time for single-GPU compute_CG: %f ms\n", elapsed.count());
            }
        }

        //copy result to CPU
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        // Compare multi-GPU result to single-GPU baseline
        bool correct = true;
        int count = 0;
        DataType max_diff = 0.0;
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            DataType diff = std::abs(result_single_GPU_h[i] - result_multi_GPU_h[i]);
            if(diff > max_diff){
                max_diff = diff;
            }
            if(std::abs(result_multi_GPU_h[i] - result_single_GPU_h[i]) > 1e-8){
                if(count<10)
                    printf("Error: CG result_multi_GPU_h != result_single_GPU_h.\t index=%d,\t result_single_GPU_h[i]=%f,\t result_multi_GPU_h[i]=%f,\t difference=%20ef\n", i, result_single_GPU_h[i], result_multi_GPU_h[i], result_single_GPU_h[i] - result_multi_GPU_h[i]);
                correct = false;
                count++;
            }
        }
        
        if(correct){
            printf("++++++\n");
            printf("CG is correct for multi GPU\n");
            printf("Max difference: %20ef\n", max_diff);
            printf("CG Multi GPU:\t n_iters_local=%d,\t normr_local=%f,\t normr0_local=%f\n", n_iters_local, normr_local, normr0_local);
            printf("CG Single GPU:\t n_iters_global=%d,\t normr_global=%20f,\t normr0_global=%.20f\n", n_iters_global, normr_global, normr0_global);
            printf("++++++\n");
        }else{
            printf("!!!!!CG Result is NOT correct!!!!!\n");
            global_int_t gn = NPX*NX*NPY*NY*NPZ*NZ;
            printf("CG: %d of %d wrong values which is %f percent\n", count, gn, (double) count/(gn)*100.0);
            printf("CG Multi GPU:\t n_iters_local=%d,\t normr_local=%f,\t normr0_local=%f\n", n_iters_local, normr_local, normr0_local);
            printf("CG Single GPU:\t n_iters_global=%d,\t normr_global=%20f,\t normr0_global=%.20f\n", n_iters_global, normr_global, normr0_global);
            printf("Difference normr_global - normr_local = %.20f\n", normr_global - normr_local);
            printf("Difference normr0_global - normr0_local = %.20f\n", normr0_global - normr0_local);
            printf("Max difference: %20ef\n", max_diff);
        }
        free(result_single_GPU_h);
        free(result_multi_GPU_h);
        free(b_global_h);
        free(x_global_h);
        CHECK_CUDA(cudaFree(b_global_d));
        CHECK_CUDA(cudaFree(x_global_d));
    }else{
        SendResult(0, halo_x_local_d, problem);
    }
}

/**
 * @brief Compares result of MG on multi-GPU to single-GPU baseline for correctness.
 */
void test_MG(striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU, striped_partial_Matrix<DataType>* A_local_striped, striped_Matrix<DataType>* A_global_striped, Halo *halo_r_local_d, Halo *halo_x_local_d, Problem *problem, bool print_timing){
    // Make sure that we work on clean data
    SetHaloGlobalIndexGPU(halo_r_local_d, problem);
    SetHaloGlobalIndexGPU(halo_x_local_d, problem);

    // Exchange halos so each process starts with the correct data
    implementation_multi_GPU.ExchangeHalo(halo_r_local_d, problem);
    implementation_multi_GPU.ExchangeHalo(halo_x_local_d, problem);

    // Run MG on multi-GPU setup
    for(int i = 0; i < (print_timing ? 10 : 1); i++){
        auto start = std::chrono::high_resolution_clock::now();
        implementation_multi_GPU.compute_MG(*A_local_striped, halo_r_local_d, halo_x_local_d, problem); //1st * 2nd = 3rd argument
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        if(print_timing){
            printf("CPU wall-clock time for multi-GPU compute_MG: %f ms\n", elapsed.count());
        }
    }

    // Verify the multi-GPU result
    if(problem->rank == 0){

        //gather result
        DataType *result_multi_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_multi_GPU_h[i] = 0;
        }
        GatherResult(halo_x_local_d, problem, result_multi_GPU_h);
        
        //compute verification result on single GPU
        DataType *result_single_GPU_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            result_single_GPU_h[i] = i;
        }
        
        //create b_global_h
        DataType *r_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            r_global_h[i] = i;
        }
        DataType *r_global_d;
        CHECK_CUDA(cudaMalloc(&r_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(r_global_d, r_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        //create x_global_h
        DataType *x_global_h = (DataType*) malloc(NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType));
        for(int i=0; i<NPX*NX*NPY*NY*NPZ*NZ; i++){
            x_global_h[i] = 0;
        }
        DataType *x_global_d;
        CHECK_CUDA(cudaMalloc(&x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType)));
        CHECK_CUDA(cudaMemcpy(x_global_d, x_global_h, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyHostToDevice));

        // Run MG on single-GPU baseline
        striped_box_coloring_Implementation<DataType> implementation_single_GPU;
        for(int i = 0; i < (print_timing ? 10 : 1); i++){
            auto start = std::chrono::high_resolution_clock::now();
            implementation_single_GPU.compute_MG(*A_global_striped, r_global_d, x_global_d);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;
            if(print_timing){
                printf("CPU wall-clock time for single-GPU compute_MG: %f ms\n", elapsed.count());
            }
        }

        //copy result to CPU
        CHECK_CUDA(cudaMemcpy(result_single_GPU_h, x_global_d, NPX*NX*NPY*NY*NPZ*NZ*sizeof(DataType), cudaMemcpyDeviceToHost));
        
        // Compare multi-GPU result to single-GPU baseline
        bool correct = true;
        int count = 0;
        DataType max_diff = 0.0;
        for(int i = 0; i < NPX*NX*NPY*NY*NPZ*NZ; i++){
            DataType diff = std::abs(result_single_GPU_h[i] - result_multi_GPU_h[i]);
            if(diff > max_diff){
                max_diff = diff;
            }
            if(std::abs(result_multi_GPU_h[i] - result_single_GPU_h[i]) > 1e-8){
                if(count<10)
                    printf("Error: MG result_multi_GPU_h != result_single_GPU_h.\t index=%d,\t result_single_GPU_h[i]=%f,\t result_multi_GPU_h[i]=%f,\t difference=%20ef\n", i, result_single_GPU_h[i], result_multi_GPU_h[i], result_single_GPU_h[i] - result_multi_GPU_h[i]);
                correct = false;
                count++;
            }
        }
        
        if(correct){
            printf("++++++\n");
            printf("MG is correct for multi GPU\n");
            printf("Max difference: %20ef\n", max_diff);
            printf("++++++\n");
        }else{
            printf("!!!!!MG Result is NOT correct!!!!!\n");
            double gn = NPX*NX*NPY*NY*NPZ*NZ;
            printf("MG: %f of %f wrong values which is %f percent\n", count, gn, count/(gn)*100.0);
            printf("Max difference: %20ef\n", max_diff);
        }
        free(result_single_GPU_h);
        free(result_multi_GPU_h);
        free(r_global_h);
        free(x_global_h);
        CHECK_CUDA(cudaFree(r_global_d));
        CHECK_CUDA(cudaFree(x_global_d));
    }else{
        SendResult(0, halo_x_local_d, problem);
    }
}

void test_cusparse(Problem *p){
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(p->nx, p->ny, p->nz);
    local_int_t * A_row_ptr_d = A.get_row_ptr_d();
    local_int_t * A_col_idx_d = A.get_col_idx_d();

}

/**
 * @brief Runs all multi-GPU validation tests and compares results to single-GPU baselines.
 */
void run_multi_GPU_tests(int argc, char *argv[], striped_multi_GPU_Implementation<DataType>& implementation_multi_GPU){

    Problem problem = *implementation_multi_GPU.init_comm(argc, argv, NPX, NPY, NPZ, NX, NY, NZ, true);

    MPI_Barrier(MPI_COMM_WORLD);
    if(problem.rank == 0){
        printf("Testing started.\n");
        printf("Comm Type: %s\n", implementation_multi_GPU.comm_type.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // initialize matrix partial matrix A_local
    striped_partial_Matrix<DataType> A_local(&problem, true, false);

    striped_partial_Matrix<DataType> *A_local_current = &A_local;
    for(int i = 0; i < 3; i++){
        A_local_current->initialize_coarse_matrix();
        A_local_current = A_local_current->get_coarse_Matrix();
    }

    // copy partial matrix to host to compare partial matrix with global matrix
    DataType *A_local_h = (DataType*) malloc(A_local.get_num_rows()*A_local.get_num_stripes()*sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(A_local_h, A_local.get_values_d(), A_local.get_num_rows()*A_local.get_num_stripes()*sizeof(DataType), cudaMemcpyDeviceToHost));

    // create global matrix for verification
    sparse_CSR_Matrix<DataType> A_global;
    A_global.generateMatrix_onGPU(NPX*NX, NPY*NY, NPZ*NZ);

    // create the coarse matrices for the MG routines
    sparse_CSR_Matrix <DataType>* current_matrix = &A_global;
    for(int i = 0; i < 3; i++){
        current_matrix->initialize_coarse_Matrix();
        current_matrix = current_matrix->get_coarse_Matrix();
    }

    striped_Matrix<DataType>* A_global_striped = A_global.get_Striped();

    // copy global matrix to host to compare partial matrix with global matrix
    DataType *A_global_h = (DataType*) malloc(A_global_striped->get_num_rows()*A_global_striped->get_num_stripes()*sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(A_global_h, A_global_striped->get_values_d(), A_global_striped->get_num_rows()*A_global_striped->get_num_stripes()*sizeof(DataType), cudaMemcpyDeviceToHost));


    // initialize p and Ap
    Halo halo_p_d;
    InitHalo(&halo_p_d, &problem);
    SetHaloGlobalIndexGPU(&halo_p_d, &problem);

    Halo halo_Ap_d;
    InitHalo(&halo_Ap_d, &problem);
    SetHaloZeroGPU(&halo_Ap_d);

    // initialize b, w, x and y
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

    // test matrix distribution
    //test_matrix_distribution(A_local.get_num_stripes(), A_global_striped->get_num_stripes(), A_local.get_num_rows(), A_global_striped->get_num_rows(), A_local_h, A_global_h, &problem);

    // test SPMV
    test_SPMV(implementation_multi_GPU, &A_local, A_global_striped, &halo_p_d, &halo_Ap_d, &problem, true);

    // test SymGS
    //test_SymGS(implementation_multi_GPU, &A_local, A_global_striped, &halo_p_d, &halo_Ap_d, &problem, false);

    // test WAXPBY
    //test_WAXPBY(implementation_multi_GPU, A_global_striped, &halo_w_d, &halo_x_d, &halo_y_d, &problem, false);

    // test Dot
    //test_Dot(implementation_multi_GPU, A_global_striped, &halo_x_d, &halo_y_d, &problem, false);

    // test CG
    //test_CG(implementation_multi_GPU, &A_local, A_global_striped, &halo_b_d, &halo_x_d, &problem, false);

    // test MG
    //test_MG(implementation_multi_GPU, &A_local, A_global_striped, &halo_b_d, &halo_x_d, &problem, false);

    // free the memory
    FreeHalo(&halo_p_d);
    FreeHalo(&halo_Ap_d);
    FreeHalo(&halo_w_d);
    FreeHalo(&halo_x_d);
    FreeHalo(&halo_y_d);
    FreeHalo(&halo_b_d);
    free(A_local_h);
    free(A_global_h);

    MPI_Barrier(MPI_COMM_WORLD);
    if(problem.rank == 0) printf("Testing done.\n", problem.rank);
    MPI_Barrier(MPI_COMM_WORLD);

    implementation_multi_GPU.finalize_comm(&problem);

}
