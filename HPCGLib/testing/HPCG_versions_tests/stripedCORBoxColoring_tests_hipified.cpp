#include "testing_hipified.hpp"


bool run_COR_BoxColoring_tests(int nx, int ny, int nz){
    bool all_pass = true;

    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    
    // create the baseline and the UUT
    striped_box_coloring_Implementation<DataType> striped_box_coloring;
    striped_COR_box_coloring_Implementation<DataType> striped_COR_box_coloring;
    
    // random seeded x vector
    std::vector<DataType> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    
    striped_Matrix<DataType>* A_striped = A.get_Striped();
    // initialize the coloring
    A_striped->generate_box_coloring();

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

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
        striped_box_coloring, striped_COR_box_coloring,
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