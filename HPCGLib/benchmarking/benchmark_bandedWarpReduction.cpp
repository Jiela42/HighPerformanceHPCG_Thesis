#include "benchmark.hpp"


void run_banded_warp_reduction_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path){

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> y = problem.second;
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);


    banded_Matrix<double> banded_A;
    banded_A.banded_Matrix_from_sparse_CSR(A);

    int num_rows = banded_A.get_num_rows();
    int num_cols = banded_A.get_num_cols();
    int nnz = banded_A.get_nnz();
    int num_bands = banded_A.get_num_bands();

    const double * matrix_data = banded_A.get_values().data();
    const int * j_min_i_data = banded_A.get_j_min_i().data();
    
    banded_warp_reduction_Implementation<double> implementation;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    double * banded_A_d;
    int * j_min_i_d;
    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&banded_A_d, num_bands * num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_bands * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(banded_A_d, matrix_data, num_bands * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_bands * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(implementation, *timer, banded_A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, a_d, b_d, x_d, y_d, result_d);

    // free the memory
    cudaFree(banded_A_d);
    cudaFree(j_min_i_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}

void run_warp_reduction_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path){
    
    banded_warp_reduction_Implementation<double> implementation;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    // the dot product is a dense operation, since we are just working on two vectors
    int nnz = nx * ny * nz;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // get two random vectors
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> y = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    // create the banded matrix
    banded_Matrix<double> A_banded;
    A_banded.set_num_rows(nx * ny * nz);

    // allocate x and y on the device
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, nx * ny * nz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, nx * ny * nz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    // run the benchmark
    bench_Dot(implementation, *timer, A_banded, x_d, y_d, result_d);

    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;    
}

void run_warp_reduction_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path){
    
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    banded_Matrix<double> banded_A;
    banded_A.banded_Matrix_from_sparse_CSR(A);

    int num_rows = banded_A.get_num_rows();
    int num_cols = banded_A.get_num_cols();
    int nnz = banded_A.get_nnz();
    int num_bands = banded_A.get_num_bands();

    const double * matrix_data = banded_A.get_values().data();
    const int * j_min_i_data = banded_A.get_j_min_i().data();
    
    banded_warp_reduction_Implementation<double> implementation;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    double * banded_A_d;
    int * j_min_i_d;
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&banded_A_d, num_bands * num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_bands * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(banded_A_d, matrix_data, num_bands * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_bands * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    bench_SPMV(implementation, *timer, banded_A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, x_d, y_d);
    // free the memory
    cudaFree(banded_A_d);
    cudaFree(j_min_i_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}

void run_warp_reduction_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path){
    
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> y = problem.second;
    std::vector<double> x(nx*ny*nz, 0.0);

    banded_Matrix<double> banded_A;
    banded_A.banded_Matrix_from_sparse_CSR(A);

    int num_rows = banded_A.get_num_rows();
    int num_cols = banded_A.get_num_cols();
    int nnz = banded_A.get_nnz();
    int num_bands = banded_A.get_num_bands();

    const double * matrix_data = banded_A.get_values().data();
    const int * j_min_i_data = banded_A.get_j_min_i().data();
    
    banded_warp_reduction_Implementation<double> implementation;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    double * banded_A_d;
    int * j_min_i_d;
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&banded_A_d, num_bands * num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_bands * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(banded_A_d, matrix_data, num_bands * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_bands * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    bench_SymGS(implementation, *timer, banded_A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, x_d, y_d);

    // free the memory
    CHECK_CUDA(cudaFree(banded_A_d));
    CHECK_CUDA(cudaFree(j_min_i_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    delete timer;
}