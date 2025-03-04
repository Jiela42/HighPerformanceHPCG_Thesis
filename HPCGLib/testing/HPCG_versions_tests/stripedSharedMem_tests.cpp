#include "testing.hpp"

#include "UtilLib/cuda_utils.hpp"

bool run_stripedSharedMem_tests_on_matrix(sparse_CSR_Matrix<double>& A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function
    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<double> cuSparse;
    Striped_Shared_Memory_Implementation<double> stripedSharedMem;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<double> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    // std::vector<double> x (nx*ny*nz, 1.0);
    // for(int i = 0; i < x.size(); i++){
    //     x[i] = i%10;
    // }

    striped_Matrix<double>* A_striped = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;

    // A_striped.striped_Matrix_from_sparse_CSR(A);

    // for(int i =0; i < A_striped.get_num_stripes(); i++){
    //     double val = A_striped.get_values()[i];
    //     if (val != 0.0){
        //     std::cout << val << std::endl;
    //     }
    // }

    // std::cout << "num_rows: " << A.get_num_rows() << std::endl;

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    int nnz = A.get_nnz();

    int num_stripes = A_striped->get_num_stripes();

    double * x_d;

    // Allocate the memory on the device

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));

    // Copy the data to the device
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    // test the SPMV function
    all_pass = all_pass && test_SPMV(
        cuSparse, stripedSharedMem,
        *A_striped,
        
        x_d
        );
    
    // anything that got allocated also needs to be de-allocted

    CHECK_CUDA(cudaFree(x_d));

    return all_pass;
}

bool run_stripedSharedMem_tests(int nx, int ny, int nz){

    bool all_pass = true;
    bool current_pass = true;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    current_pass = run_stripedSharedMem_tests_on_matrix(A);

    if(not current_pass){
        std::cout << "striped shared memory tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    all_pass = all_pass && current_pass;

    // These following aren't implemented for on GPU generation
    // A.iterative_values();

    // current_pass = run_stripedSharedMem_tests_on_matrix(A);
    // if(not current_pass){
    //     std::cout << "striped shared memory tests failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    // }

    // all_pass = all_pass && current_pass;

    // A.random_values(RANDOM_SEED);
    // current_pass = run_stripedSharedMem_tests_on_matrix(A);
    // if(not current_pass){
    //     std::cout << "striped shared memory tests failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    // }
    // all_pass = all_pass && current_pass;

    return all_pass;
   
}