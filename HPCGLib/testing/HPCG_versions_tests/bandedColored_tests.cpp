#include "testing.hpp"

#include "UtilLib/cuda_utils.hpp"

bool run_banded_colored_tests_on_matrix(sparse_CSR_Matrix<double> A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function

    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<double> cuSparse;
    banded_coloring_Implementation<double> banded_colored;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    banded_Matrix<double> A_banded;
    A_banded.banded_Matrix_from_sparse_CSR(A);

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    int num_bands = A_banded.get_num_bands();

    const int * A_row_ptr_data = A.get_row_ptr().data();
    const int * A_col_idx_data = A.get_col_idx().data();
    const double * A_values_data = A.get_values().data();

    const double * A_banded_data = A_banded.get_values().data();
    const int * j_min_i_data = A_banded.get_j_min_i().data();

    int * A_row_ptr_d;
    int * A_col_idx_d;
    double * A_values_d;

    double * banded_A_d;
    int * j_min_i_d;

    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;


    // Allocate the memory on the device
    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&a_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_cols * sizeof(double)));

    CHECK_CUDA(cudaMalloc(&banded_A_d, num_rows * num_bands * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_bands * sizeof(int)));

    // Copy the data to the device
    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(banded_A_d, A_banded_data, num_rows * num_bands * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_bands * sizeof(int), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    // test the SymGS function (minitest, does not work with banded matrices)
    all_pass = all_pass && test_SymGS(
        cuSparse, banded_colored,
        A_banded,
        A_row_ptr_d, A_col_idx_d, A_values_d,

        banded_A_d,
        num_rows, num_cols,
        num_bands,
        j_min_i_d,
        y_d
        );

    
    // anything that got allocated also needs to be de-allocted
    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));

    CHECK_CUDA(cudaFree(banded_A_d));
    CHECK_CUDA(cudaFree(j_min_i_d));

    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(a_d));
    CHECK_CUDA(cudaFree(b_d));

    return all_pass;
}

bool run_bandedColored_tests(int nx, int ny, int nz){

    bool all_pass = true;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    // create the matrix and vectors both CSR and banded
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    all_pass = all_pass && run_banded_colored_tests_on_matrix(A);

    if(not all_pass){
        std::cout << "banded_colored tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    return all_pass;
   
}