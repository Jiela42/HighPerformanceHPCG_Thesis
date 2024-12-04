#include "testing.hpp"

// since cuSparse really is our testing baseline, we only do mini-tests on cusparse
bool run_cuSparse_tests(int nx, int ny, int nz){

    // for now we just do the symGS minitest
    cuSparse_Implementation<double> cuSparse;
    sparse_CSR_Matrix<double> A;
    bool all_pass = test_SymGS(cuSparse, A);

    if (not all_pass){
        std::cout << "cuSparse SymGS mini test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    return all_pass;
}