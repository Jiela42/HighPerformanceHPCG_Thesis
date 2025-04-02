#include "testing.hpp"

#include "UtilLib/cuda_utils.hpp"

bool run_naiveStriped_tests_on_matrix(sparse_CSR_Matrix<DataType>& A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function

    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<DataType> cuSparse;
    naiveStriped_Implementation<DataType> naiveStriped;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<DataType> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    striped_Matrix<DataType> *A_striped = A.get_Striped();
  
    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();

    int num_stripes = A_striped->get_num_stripes();

    
    DataType * x_d;

    // Allocate the memory on the device
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));

    // Copy the data to the device    
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));


    // test the SPMV function
    all_pass = all_pass && test_SPMV(
        cuSparse, naiveStriped,
        *A_striped,

        x_d
        );
    
    // anything that got allocated also needs to be de-allocted
    CHECK_CUDA(cudaFree(x_d));

    return all_pass;
}

bool run_naiveStriped_tests(int nx, int ny, int nz){

    bool all_pass = true;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    // create the matrix and vectors both CSR and striped
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    all_pass = all_pass && run_naiveStriped_tests_on_matrix(A);

    if(not all_pass){
        std::cout << "naiveStriped tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    // the following tests don't work on gpu generated matrices
    // A.iterative_values();

    // all_pass = all_pass && run_naiveStriped_tests_on_matrix(A);
    // if(not all_pass){
    //     std::cout << "naiveStriped tests failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    // }

    // A.random_values(RANDOM_SEED);
    // all_pass = all_pass && run_naiveStriped_tests_on_matrix(A);
    // if(not all_pass){
    //     std::cout << "naiveStriped tests failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    // }

    return all_pass;
   
}