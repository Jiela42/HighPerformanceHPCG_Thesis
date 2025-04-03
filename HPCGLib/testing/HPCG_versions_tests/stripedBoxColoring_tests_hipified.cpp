#include "testing_hipified.hpp"

bool run_stripedBoxColoring_tests_on_Matrix(sparse_CSR_Matrix<DataType>& A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function

    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<DataType> cuSparse;
    striped_box_coloring_Implementation<DataType> striped_box_coloring;
    
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
        cuSparse, striped_box_coloring,
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

bool run_stripedBoxColoring_tests(int nx, int ny, int nz){
    bool all_pass = true;

    // create the matrix and vectors both CSR and striped
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    all_pass = all_pass && run_stripedBoxColoring_tests_on_Matrix(A);

    if(not all_pass){
        std::cout << "striped_colored tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    return all_pass;

}