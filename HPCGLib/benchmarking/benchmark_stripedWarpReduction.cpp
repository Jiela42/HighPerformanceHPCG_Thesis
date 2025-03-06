#include "benchmark.hpp"


void run_striped_warp_reduction_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation){

    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);


    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    srand(RANDOM_SEED);

    double alpha = (double)rand() / RAND_MAX;
    double beta = (double)rand() / RAND_MAX;


    if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
        // initialize the MG data
        sparse_CSR_Matrix <double>* current_matrix = &A;
        for(int i = 0; i < 3; i++){
            current_matrix->initialize_coarse_Matrix();
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    } 

    striped_Matrix<double> * striped_A = A.get_Striped();

    int num_rows = striped_A->get_num_rows();
    int num_cols = striped_A->get_num_cols();
    int nnz = striped_A->get_nnz();
    int num_stripes = striped_A->get_num_stripes();
    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(implementation, *timer, *striped_A, a_d, b_d, x_d, y_d, result_d, alpha, beta);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
    // delete striped_A;
}

void run_warp_reduction_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation){
    // get two random vectors
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> y = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    // create the striped matrix
    striped_Matrix<double> A_striped;
    A_striped.set_num_rows(nx * ny * nz);

    // allocate x and y on the device
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, nx * ny * nz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, nx * ny * nz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

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
        bench_Dot(implementation, *timer, A_striped, x_d, y_d, result_d);

        delete timer;    
    // }
    
    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);
}

void run_warp_reduction_3d27p_WAXPBY_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation){
        // get two random vectors
        std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);
        std::vector<double> y = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    
        // create the striped matrix
        striped_Matrix<double> A_striped;
        A_striped.set_num_rows(nx * ny * nz);

        srand(RANDOM_SEED);

        double alpha = (double)rand() / RAND_MAX;
        double beta = (double)rand() / RAND_MAX;
    
        // allocate x and y on the device
        double * x_d;
        double * y_d;
        double * w_d;
    
        CHECK_CUDA(cudaMalloc(&x_d, nx * ny * nz * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&y_d, nx * ny * nz * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&w_d, nx * ny * nz * sizeof(double)));
    
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(y_d, y.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

        std::string implementation_name = implementation.version_name;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        // the WAXBPY product is a dense operation, since we are just working on two vectors
        int nnz = nx * ny * nz;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);
    
        // run the benchmark
        bench_WAXPBY(implementation, *timer, A_striped, x_d, y_d, w_d, alpha, beta);

        delete timer;    
        
        // free the memory
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(w_d);
}

void run_warp_reduction_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation){
    
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);

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

    // Allocate the memory on the device
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    bench_SPMV(implementation, *timer, *striped_A, x_d, y_d);
    
    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}

void run_warp_reduction_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation){
    
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x(nx*ny*nz, 0.0);

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

    // Allocate the memory on the device
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    bench_SymGS(implementation, *timer, *striped_A, x_d, y_d);

    // free the memory
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    delete timer;
}