#include <testing.hpp>

// this tests the l2 norm calculation on CSR matrices and striped matrices
bool test_l2_norm(
    sparse_CSR_Matrix<double>& A,
    striped_Matrix<double>& striped_A,
    std::vector<double>& x_solution, std::vector<double>& y
){

    double l2_norm_host_calculated = L2_norm_for_SymGS(A, x_solution, y);

    // device calculated solution
    // put A_striped, x, y on the device

    int num_stripes = striped_A.get_num_stripes();
    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, x_solution.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, y.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));

    // calculate the l2 norm on the device

    double l2_norm_device_calculated = L2_norm_for_SymGS(
                                            striped_A,
                                            x_d, y_d
                                        );



    // // copy solution to host
    // CHECK_CUDA(cudaMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(double), cudaMemcpyHostToDevice));

    // free the memory
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    // // print norms for sanity
    // std::cout << "Host calculated l2 norm: " << l2_norm_host_calculated << std::endl;
    // std::cout << "Device calculated l2 norm: " << l2_norm_device_calculated << std::endl;

    return double_compare(l2_norm_host_calculated, l2_norm_device_calculated);

}


// this tests the l2 norm calculation on only CSR matrices
bool test_l2_norm(sparse_CSR_Matrix<double>& A, std::vector<double>& x_solution, std::vector<double>& y){
    
    // host calculated solution
    double l2_norm_host_calculated = L2_norm_for_SymGS(A, x_solution, y);

    // device calculated solution

    // copy A, x, y to device
    int * A_row_ptr_d;
    int * A_col_idx_d;
    double * A_values_d;
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, x_solution.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, y.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // calculate the l2 norm on the device
    double l2_norm_device_calculated = L2_norm_for_SymGS(
                                            A,
                                            x_d, y_d
                                        );

    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);

    // // print norms for sanity
    // std::cout << "CSR Host calculated l2 norm: " << l2_norm_host_calculated << std::endl;
    // std::cout << "CSR Device calculated l2 norm: " << l2_norm_device_calculated << std::endl;

    return double_compare(l2_norm_host_calculated, l2_norm_device_calculated);


}

bool test_rr_norm(sparse_CSR_Matrix<double>& A, std::vector<double>& x_solution, std::vector<double>& y){
    
    // host calculated solution
    double rr_norm_host_calculated = relative_residual_norm_for_SymGS(A, x_solution, y);

    // device calculated solution

    // copy x, y to device

    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, x_solution.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, y.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // calculate the l2 norm on the device
    double rr_norm_device_calculated = relative_residual_norm_for_SymGS(
                                            A,
                                            x_d, y_d
                                        );

    // free the memory

    cudaFree(x_d);
    cudaFree(y_d);

    // // print norms for sanity
    // std::cout << "CSR Host calculated rr norm: " << rr_norm_host_calculated << std::endl;
    // std::cout << "CSR Device calculated

    return double_compare(rr_norm_host_calculated, rr_norm_device_calculated);
}


bool test_rr_norm(
    sparse_CSR_Matrix<double>& A,
    striped_Matrix<double>& striped_A,
    std::vector<double>& x_solution, std::vector<double>& y
){

    double rr_norm_host_calculated = relative_residual_norm_for_SymGS(A, x_solution, y);

    // device calculated solution
    // put A_striped, x, y on the device

    int num_stripes = striped_A.get_num_stripes();
    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    double * striped_A_d;
    int * j_min_i_d;
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, x_solution.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, y.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));

    // calculate the l2 norm on the device

    double rr_norm_device_calculated = relative_residual_norm_for_SymGS(
                                            striped_A,
                                            x_d, y_d
                                        );

    // free the memory

    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    return double_compare(rr_norm_host_calculated, rr_norm_device_calculated);
    
}

bool run_all_util_tests(int nx, int ny, int nz){

    // make CSR matrix
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> y = problem.second;

    // make striped matrix
    striped_Matrix<double> striped_A;
    striped_A.striped_Matrix_from_sparse_CSR(A);

    // make fake solution
    std::vector<double> x_solution = generate_random_vector(A.get_num_rows(), 0.0, 1.0, RANDOM_SEED);

    // std::cout << "Running tests for size " << nx << "x" << ny << "x" << nz << std::endl;

    bool all_pass = true;
    all_pass = all_pass && test_l2_norm(A, x_solution, y);

    bool current_pass = test_l2_norm(A, striped_A, x_solution, y);

    if(!current_pass){
        std::cout << "l2 norm test failed for striped matrix for size " << nx << "x" << ny << "x" << nz << std::endl;
        all_pass = false;
    }

    // std::cout << "l2 norm test passed for CSR matrix for size " << nx << "x" << ny << "x" << nz << std::endl;

    current_pass = test_rr_norm(A, x_solution, y);

    if(!current_pass){
        std::cout << "rr norm test failed for CSR matrix for size " << nx << "x" << ny << "x" << nz << std::endl;
        all_pass = false;
    }

    current_pass = test_rr_norm(A, striped_A, x_solution, y);

    return all_pass;
}