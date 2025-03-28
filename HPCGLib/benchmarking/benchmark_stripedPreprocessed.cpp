#include "benchmark.hpp"


void run_striped_preprocessed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_preprocessed_Implementation<double>& implementation){

    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    std::cout << "getting striped matrix" << std::endl;
    striped_Matrix<double>* striped_A = A.get_Striped();
    // striped_A.striped_Matrix_from_sparse_CSR(A);

    int num_rows = striped_A->get_num_rows();
    int num_cols = striped_A->get_num_cols();
    int nnz = striped_A->get_nnz();
    int num_stripes = striped_A->get_num_stripes();

    // const double * matrix_data = striped_A.get_values().data();
    // const int * j_min_i_data = striped_A.get_j_min_i().data();
    
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
    bench_Implementation(implementation, *timer, *striped_A, a_d, b_d, x_d, y_d, result_d, 1.0, 1.0);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}

void run_striped_preprocessed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_preprocessed_Implementation<double>& implementation){
    
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);

    std::cout << "getting striped matrix" << std::endl;
    striped_Matrix<double>* striped_A = A.get_Striped();
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