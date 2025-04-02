#include "testing.hpp"

bool run_cuSparse_tests_on_Matrix(sparse_CSR_Matrix<DataType>& A){

    // for now we only have the dot test
    // generate a & b vectors
    std::vector<DataType> a = generate_random_vector(A.get_num_cols(), RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(A.get_num_cols(), RANDOM_SEED); 

    cuSparse_Implementation<DataType> cuSparse;

    // allocate the memory on the device
    DataType * a_d;
    DataType * b_d;

    CHECK_CUDA(cudaMalloc(&a_d, A.get_num_cols() * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&b_d, A.get_num_cols() * sizeof(DataType)));

    // copy the data to the device
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), A.get_num_cols() * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), A.get_num_cols() * sizeof(DataType), cudaMemcpyHostToDevice));

    // run the dot function
    bool test_result = test_Dot(cuSparse, A, a_d, b_d);

    // free the memory
    CHECK_CUDA(cudaFree(a_d));
    CHECK_CUDA(cudaFree(b_d));

    return test_result;
}

// since cuSparse really is our testing baseline, we only do mini-tests on cusparse
bool run_cuSparse_tests(int nx, int ny, int nz){

    // for now we just do the symGS minitest
    cuSparse_Implementation<DataType> cuSparse;
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);
    bool all_pass = true;

    if (nx == 4 && ny == 4 && nz == 4){

        sparse_CSR_Matrix<DataType> A_mini_test;

        all_pass = all_pass && test_SymGS(cuSparse, A_mini_test);

        if (not all_pass){
            std::cout << "cuSparse SymGS mini test failed" << std::endl;
        }
    }

    bool current_pass = run_cuSparse_tests_on_Matrix(A);
    if(not current_pass){
        std::cerr << "cuSparse test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
        all_pass = false;
    }


    return all_pass;
}