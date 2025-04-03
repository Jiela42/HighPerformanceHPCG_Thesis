#include "testing_hipified.hpp"

#include "UtilLib/cuda_utils_hipified.hpp"

bool run_stripedSharedMem_tests_on_matrix(sparse_CSR_Matrix<DataType>& A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function
    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<DataType> cuSparse;
    Striped_Shared_Memory_Implementation<DataType> stripedSharedMem;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<DataType> x = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    // std::vector<DataType> x (nx*ny*nz, 1.0);
    // for(int i = 0; i < x.size(); i++){
    //     x[i] = i%10;
    // }

    striped_Matrix<DataType>* A_striped = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;

    // A_striped.striped_Matrix_from_sparse_CSR(A);

    // for(int i =0; i < A_striped.get_num_stripes(); i++){
    //     DataType val = A_striped.get_values()[i];
    //     if (val != 0.0){
        //     std::cout << val << std::endl;
    //     }
    // }

    // std::cout << "num_rows: " << A.get_num_rows() << std::endl;

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();

    int num_stripes = A_striped->get_num_stripes();

    DataType * x_d;

    // Allocate the memory on the device

    CHECK_CUDA(hipMalloc(&x_d, num_cols * sizeof(DataType)));

    // Copy the data to the device
    CHECK_CUDA(hipMemcpy(x_d, x.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));

    // test the SPMV function
    all_pass = all_pass && test_SPMV(
        cuSparse, stripedSharedMem,
        *A_striped,
        
        x_d
        );
    
    // anything that got allocated also needs to be de-allocted

    CHECK_CUDA(hipFree(x_d));

    return all_pass;
}

bool run_stripedSharedMem_tests(int nx, int ny, int nz){

    bool all_pass = true;
    bool current_pass = true;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    sparse_CSR_Matrix<DataType> A;
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