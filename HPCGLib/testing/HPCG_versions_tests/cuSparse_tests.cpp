#include "testing.hpp"

bool run_cuSparse_tests_on_Matrix(sparse_CSR_Matrix<double> A){

    // for now we only have the dot test
    // generate a & b vectors
    std::vector<double> a = generate_random_vector(A.get_num_cols(), RANDOM_SEED);
    std::vector<double> b = generate_random_vector(A.get_num_cols(), RANDOM_SEED); 

    cuSparse_Implementation<double> cuSparse;

    // allocate the memory on the device
    double * a_d;
    double * b_d;

    CHECK_CUDA(cudaMalloc(&a_d, A.get_num_cols() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, A.get_num_cols() * sizeof(double)));

    // copy the data to the device
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), A.get_num_cols() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), A.get_num_cols() * sizeof(double), cudaMemcpyHostToDevice));

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
    cuSparse_Implementation<double> cuSparse;
    sparse_CSR_Matrix<double> A;
    bool all_pass = true;

    if (nx == 4 && ny == 4 && nz == 4){

        all_pass = all_pass && test_SymGS(cuSparse, A);

        if (not all_pass){
            std::cout << "cuSparse SymGS mini test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
        }
    }

    all_pass = all_pass && run_cuSparse_tests_on_Matrix(A);

    return all_pass;
}