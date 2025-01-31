#include "benchmark.hpp"


void run_striped_box_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path){
    
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> y = problem.second;
    std::vector<double> x (nx*ny*nz, 0.0);

    striped_Matrix<double> striped_A;
    striped_A.striped_Matrix_from_sparse_CSR(A);

    int num_rows = striped_A.get_num_rows();
    int num_cols = striped_A.get_num_cols();
    int nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();

    const double * matrix_data = striped_A.get_values().data();
    const int * j_min_i_data = striped_A.get_j_min_i().data();
    
    
    // Allocate the memory on the device
    double * striped_A_d;
    int * j_min_i_d;
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&striped_A_d, num_stripes * num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_stripes * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(striped_A_d, matrix_data, num_stripes * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_stripes * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    for(int i = 3; i <= 3; i ++){
        
        striped_box_coloring_Implementation<double> implementation;

        implementation.bx = i;
        implementation.by = i;
        implementation.bz = i;

        std::string box_dims = std::to_string(implementation.bx) + "x" + std::to_string(implementation.by) + "x" + std::to_string(implementation.bz);
        std::string coop_num_string = std::to_string(implementation.SymGS_cooperation_number);
        implementation.version_name = implementation.version_name + " (coloringBox: " + box_dims + ")" + " (coop_num: " + coop_num_string + ")";

        std::string implementation_name = implementation.version_name;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

        // std::cout << "Running the SymGS benchmark for the implementation: " << implementation_name << std::endl;

        bench_SymGS(implementation, *timer, striped_A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, x_d, y_d);
    
        delete timer;
    }



    // free the memory
    CHECK_CUDA(cudaFree(striped_A_d));
    CHECK_CUDA(cudaFree(j_min_i_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

}

void run_striped_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path){

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> y = problem.second;
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    striped_Matrix<double> striped_A;
    striped_A.striped_Matrix_from_sparse_CSR(A);

    int num_rows = striped_A.get_num_rows();
    int num_cols = striped_A.get_num_cols();
    int nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();

    const double * matrix_data = striped_A.get_values().data();
    const int * j_min_i_data = striped_A.get_j_min_i().data();

    
    striped_box_coloring_Implementation<double> implementation;
    // std::string box_dims = std::to_string(implementation.bx) + "x" + std::to_string(implementation.by) + "x" + std::to_string(implementation.bz);
    // std::string implementation_name = implementation.version_name + "_box: " + box_dims;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    double * striped_A_d;
    int * j_min_i_d;
    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&striped_A_d, num_stripes * num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_stripes * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(striped_A_d, matrix_data, num_stripes * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_stripes * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    // we don't want to bench the implementation for the SymGS
    // (yet, because that has a bunch of parameters, specific to that, so we do an extra call to bench_SymGS)
    implementation.SymGS_implemented = false;
    bench_Implementation(implementation, *timer, striped_A, striped_A_d, num_rows, num_cols, num_stripes, j_min_i_d, a_d, b_d, x_d, y_d, result_d);

    // free the memory
    cudaFree(striped_A_d);
    cudaFree(j_min_i_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;

    // run the SymGS benchmark
    run_striped_box_coloring_3d27p_SymGS_benchmark(nx, ny, nz, folder_path);
}
