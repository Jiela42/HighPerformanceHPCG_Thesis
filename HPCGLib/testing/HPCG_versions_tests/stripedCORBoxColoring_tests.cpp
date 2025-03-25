#include "testing.hpp"


bool run_COR_BoxColoring_tests(int nx, int ny, int nz){
    bool all_pass = true;

    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    
    // create the baseline and the UUT
    striped_box_coloring_Implementation<double> striped_box_coloring;
    striped_COR_box_coloring_Implementation<double> striped_COR_box_coloring;
    
    // random seeded x vector
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    
    striped_Matrix<double>* A_striped = A.get_Striped();
    // initialize the coloring
    A_striped->generate_box_coloring();

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
        striped_box_coloring, striped_COR_box_coloring,
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