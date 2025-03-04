#include <testing.hpp>

bool run_no_store_stripedColoring_tests_onMatrix(sparse_CSR_Matrix<double>& A){
    
    // I believe this method only has SymGS
    bool all_pass = true;

    // create the baseline and the UUT
    cuSparse_Implementation<double> cuSparse;
    no_store_striped_coloring_Implementation<double> no_store_implementation;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    striped_Matrix<double>* A_striped = A.get_Striped();
    // A_striped.striped_Matrix_from_sparse_CSR(A);
    std::cout << "getting striped matrix" << std::endl;

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    int num_stripes = A_striped->get_num_stripes();

    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;

    // Allocate the memory on the device
    CHECK_CUDA(cudaMalloc(&a_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_cols * sizeof(double)));

    // Copy the data to the device    
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    // test the SymGS function (minitest, does not work with striped matrices)
    all_pass = all_pass && test_SymGS(
        cuSparse, no_store_implementation,
        *A_striped,

        y_d
        );

    
    // anything that got allocated also needs to be de-allocted

    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(a_d));
    CHECK_CUDA(cudaFree(b_d));

    return all_pass;
}

bool run_no_store_stripedColoring_filebased_tests(){
    
    bool all_pass = true;

    no_store_striped_coloring_Implementation<double> no_store_implementation;

    // MG tests
    all_pass = all_pass && test_MG(no_store_implementation);

    std::cout << "Finished MG tests" << std::endl;

    // CG tests
    // all_pass = all_pass && test_CG(no_store_implementation);

    return all_pass;
}

bool run_no_store_stripedColoring_tests(int nx, int ny, int nz) {
    bool all_pass = true;

    // create the matrix and vectors both CSR and striped
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    all_pass = all_pass && run_no_store_stripedColoring_tests_onMatrix(A);

    return all_pass;

}