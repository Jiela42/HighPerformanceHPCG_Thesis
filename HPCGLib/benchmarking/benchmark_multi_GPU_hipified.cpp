#include "benchmark_hipified.hpp"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"
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
    CHECK_CUDA(hipMalloc(&y_vector_d, n * sizeof(DataType)));
    generate_y_vector_for_HPCG_problem_onGPU(problem, y_vector_d);
    InjectDataToHalo(&y, y_vector_d);
    CHECK_CUDA(hipFree(y_vector_d));
    
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
    CHECK_CUDA(hipMalloc(&result_d, sizeof(DataType)));
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
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
 
    MPITimer* timer = new MPITimer(nx, ny, nz, 0, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // run the benchmarks
    bench_Implementation(implementation, *timer, A, &a, &b, &x, &y, result_d, alpha, beta, problem, benchFilter);

    // free the memory
    FreeHalo(&y);
    FreeHalo(&x);
    FreeHalo(&a);
    FreeHalo(&b);
    CHECK_CUDA(hipFree(result_d));

    delete timer;
}


/*
void run_multi_GPU_Dot_benchmark(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_multi_GPU_Implementation<DataType>& implementation){

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

    //set Device
    InitGPU(&problem);

    //Initialize data
    Halo y;
    InitHalo(&y, nx, ny, nz);
    SetHaloRandomGPU(&y, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);
    
    Halo x;
    InitHalo(&x, nx, ny, nz);
    SetHaloRandomGPU(&x, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);

    DataType *result_d;
    CHECK_CUDA(hipMalloc(&result_d, sizeof(DataType)));

    // create the striped matrix
    striped_Matrix<DataType> A_striped;
    A_striped.set_num_rows(nx * ny * nz);


    // for(int i = 8; i <= 8; i=i << 1){
        // std::cout << "Cooperation number = " << i << std::endl;

        std::string implementation_name = implementation.version_name;
        // implementation.dot_cooperation_number = i;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        // the dot product is a dense operation, since we are just working on two vectors
        int nnz = nx * ny * nz;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);
    
        // run the benchmark
        bench_Dot(implementation, *timer, A, &x, &y, &result_d);

        delete timer;    
    // }
    
    // free the memory
    FreeHalo(&y);
    FreeHalo(&x);
    hipFree(result_d);
}

void run_multi_GPU_WAXPBY_benchmark(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_multi_GPU_Implementation<DataType>& implementation){
        
        int size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        //initialize problem struct
        Problem problem; //holds geometric data about the problem
        GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

        //set Device
        InitGPU(&problem);

        //Initialize data
        Halo y;
        InitHalo(&y, nx, ny, nz);
        SetHaloRandomGPU(&y, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);
        
        Halo x;
        InitHalo(&x, nx, ny, nz);
        SetHaloRandomGPU(&x, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);

        Halo w;
        InitHalo(&w, nx, ny, nz);
        SetHaloRandomGPU(&w, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);
        
        srand(RANDOM_SEED);

        double alpha = (double)rand() / RAND_MAX;
        double beta = (double)rand() / RAND_MAX;

        std::string implementation_name = implementation.version_name;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        // the WAXBPY product is a dense operation, since we are just working on two vectors
        int nnz = nx * ny * nz;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);
    
        // run the benchmark
        bench_WAXPBY(implementation, *timer, A_striped, &x, &y, &w, alpha, beta, &problem);

        delete timer;    
        
        // free the memory
        FreeHalo(&y);
        FreeHalo(&x);
        FreeHalo(&w);
}

void run_multi_GPU_SPMV_benchmark(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

    //set Device
    InitGPU(&problem);

    //Initialize data
    Halo y;
    InitHalo(&y, nx, ny, nz);
    SetHaloZeroGPU(&y, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);
    
    Halo x;
    InitHalo(&x, nx, ny, nz);
    SetHaloRandomGPU(&x, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);

    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    striped_Matrix<DataType>* striped_A = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;
    // striped_A.striped_Matrix_from_sparse_CSR(A);

    int num_rows = striped_A->get_num_rows();
    int num_cols = striped_A->get_num_cols();
    int nnz = striped_A->get_nnz();
    int num_stripes = striped_A->get_num_stripes();
    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    bench_SPMV(implementation, *timer, *striped_A, &x, &y, &problem);
    
    // free the memory
    FreeHalo(&y);
    FreeHalo(&x);

    delete timer;
}

void run_multi_GPU_SymGS_benchmark(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

    //set Device
    InitGPU(&problem);

    //Initialize data
    Halo y;
    InitHalo(&y, nx, ny, nz);
    GeneratePartialYVectorForHPCGProblem(nx, ny, nz, &y);

    
    Halo x;
    InitHalo(&x, nx, ny, nz);
    SetHaloRandomGPU(&x, problem, RANDOM_MIN, RANDOM_MAX, RANDOM_SEED);

    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    striped_Matrix<double>* striped_A = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;

    // striped_A.striped_Matrix_from_sparse_CSR(A);

    int num_rows = striped_A->get_num_rows();
    int num_cols = striped_A->get_num_cols();
    int nnz = striped_A->get_nnz();
    int num_stripes = striped_A->get_num_stripes();
    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    bench_SymGS(implementation, *timer, *striped_A, &x, &y, &problem);

    // free the memory
    FreeHalo(&y);
    FreeHalo(&x);

    delete timer;
}

void run_multi_GPU_Exchange_Halo_benchmark(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //initialize problem struct
    Problem problem; //holds geometric data about the problem
    GenerateProblem(npx, npy, npz, nx, ny, nz, size, rank, &problem);

    //set Device
    InitGPU(&problem);
    
    Halo x;
    InitHalo(&x, nx, ny, nz);
    SetHaloGlobalIndexGPU(&x, problem);
    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    cpuTimer* timer = new cpuTimer(nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    bench_ExchangeHalo(implementation, *timer, &x, &problem);

    // free the memory
    FreeHalo(&x);

    delete timer;
}
*/