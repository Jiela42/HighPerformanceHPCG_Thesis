#include "benchmark.hpp"

void run_bandedSharedMem_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path){

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
    
    Banded_Shared_Memory_Implementation<double> implementation;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, "41-44", "3d_27pt", implementation_name, additional_params, folder_path);

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

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(implementation, *timer, banded_A, banded_A_d, num_rows, num_cols, num_bands, j_min_i_d, x_d, y_d, result_d);

    // free the memory
    cudaFree(banded_A_d);
    cudaFree(j_min_i_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}