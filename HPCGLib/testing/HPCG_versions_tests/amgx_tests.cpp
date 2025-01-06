#include "testing.hpp"
bool run_amgx_tests_on_matrix(sparse_CSR_Matrix<double> A){
    // these tests run on any matrix because the methods tested are not matrix dependent (SPMV)
    
    // for now this is all just a csr format

    bool all_pass = true;

    // create the baseline and the UUT
    cuSparse_Implementation<double> cuSparse;
    amgx_Implementation<double> amgx;

    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // random seeded x vector
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    // allocate the memory on the device
    int * A_row_ptr_d;
    int * A_col_idx_d;
    double * A_values_d;
    double * x_d;

    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, A.get_nnz() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, A.get_nnz() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));

    // copy the data to the device
    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A.get_row_ptr().data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A.get_col_idx().data(), A.get_nnz() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A.get_values().data(), A.get_nnz() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));


    // test the SPMV function
    bool test_pass = test_SPMV(
        cuSparse, amgx,
        A,
        A_row_ptr_d, A_col_idx_d, A_values_d,
        x_d);
    
    if(not test_pass){
        all_pass = false;
        std::cout << "AMGX SPMV test failed" << std::endl;
    }

    // free the memory
    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));
    CHECK_CUDA(cudaFree(x_d));
    
    return all_pass;
}

bool run_amgx_HPCG_tests(sparse_CSR_Matrix<double> A, std::vector<double> y){
    // these tests require the standard unchanged HPCG matrix
    // the vector is what is the y used for the SymGS
    return true;
}


bool run_amgx_tests(int nx, int ny, int nz){

    bool all_pass = true;
    bool test_pass = true;
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;
    std::vector<double> y = problem.second;

    test_pass = test_pass && run_amgx_HPCG_tests(A,y);
    test_pass = test_pass && run_amgx_tests_on_matrix(A);

    if(not test_pass){
        all_pass = false;
        std::cout << "AMGX tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    A.iterative_values();

    test_pass = run_amgx_tests_on_matrix(A);

    if(not test_pass){
        all_pass = false;
        std::cout << "AMGX tests failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    A.random_values(RANDOM_SEED);

    test_pass = run_amgx_tests_on_matrix(A);

    if(not test_pass){
        all_pass = false;
        std::cout << "AMGX tests failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    return all_pass;
}