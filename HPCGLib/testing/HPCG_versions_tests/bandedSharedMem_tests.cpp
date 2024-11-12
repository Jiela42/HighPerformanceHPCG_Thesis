#include "testing.hpp"

#include "UtilLib/cuda_utils.hpp"

bool run_bandedSharedMem_tests_on_matrix(sparse_CSR_Matrix<double> A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function
    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<double> cuSparse;
    Banded_Shared_Memory_Implementation<double> bandedSharedMem;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    // std::vector<double> x (nx*ny*nz, 1.0);
    // for(int i = 0; i < x.size(); i++){
    //     x[i] = i%10;
    // }

    banded_Matrix<double> A_banded;
    A_banded.banded_Matrix_from_sparse_CSR(A);

    // for(int i =0; i < A_banded.get_num_bands(); i++){
    //     double val = A_banded.get_values()[i];
    //     if (val != 0.0){
        //     std::cout << val << std::endl;
    //     }
    // }

    std::cout << "num_rows: " << A.get_num_rows() << std::endl;

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

    double * x_d;

    // Allocate the memory on the device
    CHECK_CUDA(cudaMalloc(&A_row_ptr_d, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_values_d, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));

    CHECK_CUDA(cudaMalloc(&banded_A_d, num_rows * num_bands * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_bands * sizeof(int)));

    // Copy the data to the device
    CHECK_CUDA(cudaMemcpy(A_row_ptr_d, A_row_ptr_data, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_col_idx_d, A_col_idx_data, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_values_d, A_values_data, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(banded_A_d, A_banded_data, num_rows * num_bands * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, j_min_i_data, num_bands * sizeof(int), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    // test the SPMV function
    all_pass = all_pass && test_SPMV(
        cuSparse, bandedSharedMem,
        A_banded,
        A_row_ptr_d, A_col_idx_d, A_values_d,
        
        banded_A_d,
        num_rows, num_cols,
        num_bands,
        j_min_i_d,

        x_d
        );
    
    // anything that got allocated also needs to be de-allocted
    CHECK_CUDA(cudaFree(A_row_ptr_d));
    CHECK_CUDA(cudaFree(A_col_idx_d));
    CHECK_CUDA(cudaFree(A_values_d));

    CHECK_CUDA(cudaFree(banded_A_d));
    CHECK_CUDA(cudaFree(j_min_i_d));

    CHECK_CUDA(cudaFree(x_d));

    return all_pass;
}

bool run_bandedSharedMem_tests(int nx, int ny, int nz){

    bool all_pass = true;
    bool current_pass = true;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    // create the matrix and vectors both CSR and banded
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    current_pass = run_bandedSharedMem_tests_on_matrix(A);

    if(not current_pass){
        std::cout << "banded shared memory tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    all_pass = all_pass && current_pass;

    A.iterative_values();

    current_pass = run_bandedSharedMem_tests_on_matrix(A);
    if(not current_pass){
        std::cout << "banded shared memory tests failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    all_pass = all_pass && current_pass;

    A.random_values(RANDOM_SEED);
    current_pass = run_bandedSharedMem_tests_on_matrix(A);
    if(not current_pass){
        std::cout << "banded shared memory tests failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    all_pass = all_pass && current_pass;

    return all_pass;
   
}