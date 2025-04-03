#include <testing_hipified.hpp>

// this tests the l2 norm calculation on CSR matrices and striped matrices
bool test_l2_norm(
    sparse_CSR_Matrix<DataType>& A,
    striped_Matrix<DataType>& striped_A,
    std::vector<DataType>& x_solution, std::vector<DataType>& y
){

    double l2_norm_host_calculated = L2_norm_for_SymGS(A, x_solution, y);

    // device calculated solution
    // put A_striped, x, y on the device

    int num_stripes = striped_A.get_num_stripes();
    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();

    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(hipMalloc(&x_d, x_solution.size() * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, y.size() * sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), y.size() * sizeof(DataType), hipMemcpyHostToDevice));

    // calculate the l2 norm on the device
    // we made this a function of the HPCGLib of the implementation, so we need an instance of an implementation
    striped_box_coloring_Implementation<DataType> impl;

    double l2_norm_device_calculated = impl.L2_norm_for_SymGS(
                                            striped_A,
                                            x_d, y_d
                                        );



    // // copy solution to host
    // CHECK_CUDA(hipMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(double), hipMemcpyHostToDevice));

    // free the memory
    CHECK_CUDA(hipFree(x_d));
    CHECK_CUDA(hipFree(y_d));

    // // print norms for sanity
    std::cout << "Host calculated l2 norm: " << l2_norm_host_calculated << std::endl;
    std::cout << "Device calculated l2 norm: " << l2_norm_device_calculated << std::endl;
    return relaxed_double_compare(l2_norm_host_calculated, l2_norm_device_calculated, 1e-10);

}


// this tests the l2 norm calculation on only CSR matrices
bool test_l2_norm(sparse_CSR_Matrix<DataType>& A, std::vector<DataType>& x_solution, std::vector<DataType>& y){
    
    // host calculated solution
    double l2_norm_host_calculated = L2_norm_for_SymGS(A, x_solution, y);

    // device calculated solution

    // copy A, x, y to device
    local_int_t * A_row_ptr_d;
    local_int_t * A_col_idx_d;
    DataType * A_values_d;
    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(hipMalloc(&x_d, x_solution.size() * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, y.size() * sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), y.size() * sizeof(DataType), hipMemcpyHostToDevice));

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();

    // calculate the l2 norm on the device
    double l2_norm_device_calculated = L2_norm_for_SymGS(
                                            A,
                                            x_d, y_d
                                        );

    // free the memory
    hipFree(x_d);
    hipFree(y_d);

    // // print norms for sanity
    std::cout << "CSR Host calculated l2 norm: " << l2_norm_host_calculated << std::endl;
    std::cout << "CSR Device calculated l2 norm: " << l2_norm_device_calculated << std::endl;

    return (l2_norm_host_calculated, l2_norm_device_calculated);


}

bool test_rr_norm(sparse_CSR_Matrix<DataType>& A, std::vector<DataType>& x_solution, std::vector<DataType>& y){
    
    // host calculated solution
    double rr_norm_host_calculated = relative_residual_norm_for_SymGS(A, x_solution, y);

    // device calculated solution

    // copy x, y to device

    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(hipMalloc(&x_d, x_solution.size() * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, y.size() * sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), y.size() * sizeof(DataType), hipMemcpyHostToDevice));

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();

    // calculate the l2 norm on the device
    double rr_norm_device_calculated = relative_residual_norm_for_SymGS(
                                            A,
                                            x_d, y_d
                                        );

    // free the memory

    hipFree(x_d);
    hipFree(y_d);

    // // print norms for sanity
    // std::cout << "CSR Host calculated rr norm: " << rr_norm_host_calculated << std::endl;
    // std::cout << "CSR Device calculated

    return double_compare(rr_norm_host_calculated, rr_norm_device_calculated);
}


bool test_rr_norm(
    sparse_CSR_Matrix<DataType>& A,
    striped_Matrix<DataType>& striped_A,
    std::vector<DataType>& x_solution, std::vector<DataType>& y
){

    double rr_norm_host_calculated = relative_residual_norm_for_SymGS(A, x_solution, y);

    // device calculated solution
    // put A_striped, x, y on the device

    int num_stripes = striped_A.get_num_stripes();
    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();

    DataType * striped_A_d;
    local_int_t * j_min_i_d;
    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(hipMalloc(&x_d, x_solution.size() * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, y.size() * sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(x_d, x_solution.data(), x_solution.size() * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), y.size() * sizeof(DataType), hipMemcpyHostToDevice));

    // calculate the l2 norm on the device

    double rr_norm_device_calculated = relative_residual_norm_for_SymGS(
                                            striped_A,
                                            x_d, y_d
                                        );

    // free the memory

    CHECK_CUDA(hipFree(x_d));
    CHECK_CUDA(hipFree(y_d));

    return relaxed_double_compare(rr_norm_host_calculated, rr_norm_device_calculated, 1e-10);
    
}

bool run_all_util_tests(int nx, int ny, int nz){

    // make CSR matrix
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    // std::cout << "CSR Matrix is generated" << std::endl;
    // make striped matrix
    std::cout << "getting striped matrix" << std::endl;
    striped_Matrix<DataType>* striped_A = A.get_Striped();

    // make fake solution
    std::vector<DataType> x_solution = generate_random_vector(A.get_num_rows(), 0.0, 1.0, RANDOM_SEED);

    // std::cout << "Running tests for size " << nx << "x" << ny << "x" << nz << std::endl;

    bool all_pass = true;
    all_pass = all_pass && test_l2_norm(A, x_solution, y);

    // std::cout<< "l2 norm test passed for CSR matrix for size " << nx << "x" << ny << "x" << nz << std::endl;

    bool current_pass = test_l2_norm(A, *striped_A, x_solution, y);

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

    current_pass = test_rr_norm(A, *striped_A, x_solution, y);

    return all_pass;
}