#include "benchmark.hpp"

void run_naiveStriped_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, naiveStriped_Implementation<DataType>& implementation){

    striped_Matrix<DataType> striped_A;
    striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);

    local_int_t num_rows = striped_A.get_num_rows();
    local_int_t num_cols = striped_A.get_num_cols();
    local_int_t nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();

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
    DataType * result_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(implementation, *timer, striped_A, a_d, b_d, x_d, y_d, result_d, 1.0, 1.0);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;

}