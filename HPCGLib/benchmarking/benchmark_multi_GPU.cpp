#include "benchmark.hpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "mpi.h"

#define RANDOM_MIN 0
#define RANDOM_MAX 1000

void run_multi_GPU_benchmarks(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_multi_GPU_Implementation<DataType>& implementation, Problem *problem, const std::string& benchFilter){

    local_int_t n = nx * ny * nz;

    //initialize matrix partial matrix A_local
    striped_partial_Matrix<DataType> A(problem);

    //Initialize data
    Halo y;
    InitHalo(&y, problem);
    DataType *y_vector_d;
    CHECK_CUDA(cudaMalloc(&y_vector_d, n * sizeof(DataType)));
    generate_y_vector_for_HPCG_problem_onGPU(problem, y_vector_d);
    InjectDataToHalo(&y, y_vector_d);
    CHECK_CUDA(cudaFree(y_vector_d));
    
    Halo x;
    InitHalo(&x, problem);

    Halo a;
    InitHalo(&a, problem);
    SetHaloRandomGPU(&a, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);

    Halo b;
    InitHalo(&b, problem);
    SetHaloRandomGPU(&b, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);

    srand(RANDOM_SEED);

    DataType *result_d;
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(DataType)));
    DataType alpha = (DataType)rand() / RAND_MAX;
    DataType beta = (DataType)rand() / RAND_MAX;

    if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
        // initialize the MG data
        striped_partial_Matrix<DataType>* current_matrix = &A;
        for(int i = 0; i < 3; i++){
            current_matrix->initialize_coarse_matrix();
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    } 

    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters + " NPX=" + std::to_string(npx) + " NPY=" + std::to_string(npy) + " NPZ=" + std::to_string(npz);
    std::string ault_node = implementation.ault_nodes;
 
    MPITimer* timer = new MPITimer(nx, ny, nz, 0, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // run the benchmarks
    bench_Implementation(implementation, *timer, A, &a, &b, &x, &y, result_d, alpha, beta, problem, benchFilter);

    // free the memory
    FreeHalo(&y);
    FreeHalo(&x);
    FreeHalo(&a);
    FreeHalo(&b);
    CHECK_CUDA(cudaFree(result_d));

    delete timer;
}
