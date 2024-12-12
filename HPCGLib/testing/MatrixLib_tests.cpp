#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "testing.hpp"


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
bool read_save_test(striped_Matrix<double> A, std::string info){
    // std::string str_nx = std::to_string(A.get_nx());
    // std::string str_ny = std::to_string(A.get_ny());
    // std::string str_nz = std::to_string(A.get_nz());

    // A.write_to_file();
    // striped_Matrix<double> A_from_file;
    // A_from_file.read_from_file(str_nx, str_ny, str_nz, "cpp");

    // bool test_passed = A.compare_to(A_from_file, info);
    // if (not test_passed){
    //     std::cerr << "Test failed for " << info << std::endl;
    // }
}

//////////////////////////////////////////////////////////////////////////////////////////////
// striped vs csr tests
bool striped_csr_conversion_test_on_matrix(sparse_CSR_Matrix<double> A){
    striped_Matrix<double> striped_A;
    striped_A.striped_Matrix_from_sparse_CSR(A);
    // for(int i = 0; i < striped_A.get_num_rows(); i++){
    //     for(int j = 0; j < striped_A.get_num_cols(); j++){
    //         double elem = A.get_element(i, j);
    //         double striped_elem = striped_A.get_element(i, j);
    //         if (elem != striped_elem){
    //             std::cerr << "element mismatch at i: " << i << " j: " << j << " elem: " << elem << " striped_elem: " << striped_elem << std::endl;
    //             return false;
    //         }
    //     }
    // }

    // for(int i=0; i< striped_A.get_num_stripes(); i++){
    //     double val = striped_A.get_values()[i];
    //     std::cout << "in test val: " << val << std::endl;
    // }
    sparse_CSR_Matrix<double> back_to_CSR;
    // std::cout << "in test striped A nnz: " << striped_A.get_nnz() << std::endl;
    back_to_CSR.sparse_CSR_Matrix_from_striped(striped_A);

    bool test_passed = A.compare_to(back_to_CSR, "striped vs csr conversion test");

    return test_passed;
}

bool run_striped_csr_conversion_test(int nx, int ny, int nz){
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    bool all_pass = true;

    all_pass = all_pass && striped_csr_conversion_test_on_matrix(A);
    if (not all_pass){
        std::cerr << "striped vs csr conversion test failed for normal HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    // std::cout << "random values" << std::endl;
    A.random_values(RANDOM_SEED);
    all_pass = all_pass && striped_csr_conversion_test_on_matrix(A);
    if (not all_pass){
        std::cerr << "striped vs csr conversion test failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }   
    // std::cout << "iterative values" << std::endl;
    A.iterative_values();
    all_pass = all_pass && striped_csr_conversion_test_on_matrix(A);
    if (not all_pass){
        std::cerr << "striped vs csr conversion test failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    return all_pass;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// for each size we recieve we generate the matrices and run all the tests

bool run_all_matrixLib_tests(int nx, int ny, int nz){
    
    std::string dim_info = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);

    // run tests on standard 3d27p matrices
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    striped_Matrix<double> striped_A;
    striped_A.striped_Matrix_from_sparse_CSR(A);

    bool all_pass = true;

    all_pass = all_pass && read_save_test(A, "read_save_test on 3d27p CSR Matrix for " + dim_info);
    // read_save_test(striped_A, "read_save_test on 3d27p striped Matrix for " + dim_info);
    all_pass = all_pass && run_striped_csr_conversion_test(nx, ny, nz);

    return all_pass;

}