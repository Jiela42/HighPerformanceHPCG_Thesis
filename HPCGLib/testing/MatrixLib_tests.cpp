#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/banded_Matrix.hpp"


//////////////////////////////////////////////////////////////////////////////////////////////
// we have the single test files
//////////////////////////////////////////////////////////////////////////////////////////////

// read_save_tests
bool read_save_test(sparse_CSR_Matrix<double> A, std::string info){
    std::string str_nx = std::to_string(A.get_nx());
    std::string str_ny = std::to_string(A.get_ny());
    std::string str_nz = std::to_string(A.get_nz());

    A.write_to_file();
    sparse_CSR_Matrix<double> A_from_file;
    A_from_file.read_from_file(str_nx, str_ny, str_nz, "cpp");

    bool test_passed = A.compare_to(A_from_file, info);

    if (not test_passed){
        std::cerr << "read_save_test failed for " << info << std::endl;
    }
    return test_passed;
    
}

// this does not yet exist, but once it does, we need to implement it
bool read_save_test(banded_Matrix<double> A, std::string info){
    // std::string str_nx = std::to_string(A.get_nx());
    // std::string str_ny = std::to_string(A.get_ny());
    // std::string str_nz = std::to_string(A.get_nz());

    // A.write_to_file();
    // banded_Matrix<double> A_from_file;
    // A_from_file.read_from_file(str_nx, str_ny, str_nz, "cpp");

    // bool test_passed = A.compare_to(A_from_file, info);
    // if (not test_passed){
    //     std::cerr << "Test failed for " << info << std::endl;
    // }
}

//////////////////////////////////////////////////////////////////////////////////////////////
// banded vs csr tests
bool run_banded_csr_conversion_test(int nx, int ny, int nz){
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    banded_Matrix<double> banded_A;
    banded_A.banded_Matrix_from_sparse_CSR(A);
    sparse_CSR_Matrix<double> back_to_CSR;
    back_to_CSR.sparse_CSR_Matrix_from_banded(banded_A);

    bool test_passed = A.compare_to(back_to_CSR, "banded vs csr conversion test for " + std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz));
    
    if (not test_passed){
        std::cerr << "Test failed for banded vs csr conversion test for " << nx << "x" << ny << "x" << nz << std::endl;
    }

    return test_passed;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// for each size we recieve we generate the matrices and run all the tests

bool run_all_matrixLib_tests(int nx, int ny, int nz){
    
    std::string dim_info = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);

    // run tests on standard 3d27p matrices
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    banded_Matrix<double> banded_A;
    banded_A.banded_Matrix_from_sparse_CSR(A);

    bool all_pass = true;

    all_pass = all_pass && read_save_test(A, "read_save_test on 3d27p CSR Matrix for " + dim_info);
    // read_save_test(banded_A, "read_save_test on 3d27p banded Matrix for " + dim_info);
    all_pass = all_pass && run_banded_csr_conversion_test(nx, ny, nz);

    return all_pass;

}