#include "benchmark.hpp"

void run_stripedSharedMem_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path){

    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    striped_Matrix<double> striped_A;
    striped_A.striped_Matrix_from_sparse_CSR(A);

    int num_rows = striped_A.get_num_rows();
    int num_cols = striped_A.get_num_cols();
    int nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();

    
    Striped_Shared_Memory_Implementation<double> implementation;
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
    bench_Implementation(implementation, *timer, striped_A, a_d, b_d, x_d, y_d, result_d);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}