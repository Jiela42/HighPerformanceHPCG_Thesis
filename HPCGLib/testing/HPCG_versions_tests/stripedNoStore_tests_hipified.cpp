#include <testing_hipified.hpp>

bool run_no_store_stripedColoring_tests_onMatrix(sparse_CSR_Matrix<DataType>& A){
    
    // I believe this method only has SymGS
    bool all_pass = true;

    // create the baseline and the UUT
    cuSparse_Implementation<DataType> cuSparse;
    no_store_striped_coloring_Implementation<DataType> no_store_implementation;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<DataType> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    striped_Matrix<DataType>* A_striped = A.get_Striped();

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();

    int num_stripes = A_striped->get_num_stripes();

    DataType * a_d;
    DataType * b_d;
    DataType * x_d;
    DataType * y_d;

    // Allocate the memory on the device
    CHECK_CUDA(hipMalloc(&a_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&b_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, num_cols * sizeof(DataType)));

    // Copy the data to the device    
    CHECK_CUDA(hipMemcpy(a_d, a.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(b_d, b.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));

    // test the SymGS function (minitest, does not work with striped matrices)
    all_pass = all_pass && test_SymGS(
        cuSparse, no_store_implementation,
        *A_striped,

        y_d
        );

    
    // anything that got allocated also needs to be de-allocted

    CHECK_CUDA(hipFree(x_d));
    CHECK_CUDA(hipFree(y_d));
    CHECK_CUDA(hipFree(a_d));
    CHECK_CUDA(hipFree(b_d));

    return all_pass;
}

bool run_no_store_stripedColoring_filebased_tests(){
    
    bool all_pass = true;

    no_store_striped_coloring_Implementation<DataType> no_store_implementation;

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
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    all_pass = all_pass && run_no_store_stripedColoring_tests_onMatrix(A);

    return all_pass;

}