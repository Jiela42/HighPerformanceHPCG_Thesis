#include "benchmark.hpp"

void run_cuSparse_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation){

    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();

    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<DataType> x (num_rows, 0.0);
    std::vector<DataType> a = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(num_rows, RANDOM_SEED);
    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * a_d;
    DataType * b_d;
    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(implementation, *timer, A, a_d, b_d, x_d, y_d);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);

    delete timer;
}

void run_cuSparse_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation){

    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();
    
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    // std::vector<DataType> x = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> x (num_rows, 0.0);

    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    bench_SymGS(implementation, *timer, A, x_d, y_d);

    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);

    delete timer;
}

void run_cuSparse_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation){

    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();

    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<DataType> a = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(num_rows, RANDOM_SEED);
    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * a_d;
    DataType * y_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    bench_SPMV(implementation, *timer, A,  a_d, y_d);

    // free the memory
    cudaFree(a_d);
    cudaFree(y_d);

    delete timer;
}

void run_cuSparse_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation){

    // generate matrix
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    local_int_t num_rows = A.get_num_rows();

    // generate x & y vectors
    std::vector<DataType> x = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> y = generate_random_vector(num_rows, RANDOM_SEED);

    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, num_rows, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * x_d;
    DataType * y_d;
    DataType * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    bench_Dot(implementation, *timer, A, x_d, y_d, result_d);

    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;

}

