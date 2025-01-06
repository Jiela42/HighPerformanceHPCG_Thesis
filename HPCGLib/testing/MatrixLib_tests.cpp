#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "MatrixLib/coloring.cuh"
#include "testing.hpp"
#include <algorithm>


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
    return true;
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

bool coloring_test(striped_Matrix<double> A){

    bool all_pass = true;

    // first we get the long hand computed colors
    std::vector<int> colors_dynamically = color_for_forward_pass(A);

    // then we get the colors from the matrix
    std::vector<int> colors_statically (A.get_num_rows(), 0);

    for (int i = 0 ; i < A.get_num_rows(); i++){
        int x = i % A.get_nx();
        int y = (i / A.get_nx()) % A.get_ny();
        int z = i / (A.get_nx() * A.get_ny());
        colors_statically[i] = x + 2*y + 4*z;
    }

    bool colors_stat_vs_dym_test = vector_compare(colors_statically, colors_dynamically, "coloring test");
    if (not colors_stat_vs_dym_test){
        std::cerr << "coloring static vs dynamic test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }
    all_pass = all_pass && colors_stat_vs_dym_test;

    int max_col_dyn = *std::max_element(colors_dynamically.begin(), colors_dynamically.end());
    int max_col_stat = *std::max_element(colors_statically.begin(), colors_statically.end());

    // std::cout << "num colors " << max_col_dyn + 1 << std::endl;

    std::cout << "size: " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;


    std::vector<int> color_ptr_dyn(max_col_dyn+2, 0);
    std::vector<int> color_ptr_stat(max_col_stat+2, 0);

    for(int i = 0; i <= max_col_dyn; i++){
        int color_ct_i = count(colors_dynamically.begin(), colors_dynamically.end(), i);
        color_ptr_dyn[i+1] = color_ptr_dyn[i] + count(colors_dynamically.begin(), colors_dynamically.end(), i);
        color_ptr_stat[i+1] = color_ptr_dyn[i] + std::count(colors_statically.begin(), colors_statically.end(), i);
        std::cout << "color " << i << " has " << color_ct_i << " rows" << std::endl;
    }

    bool color_ptr_stat_vs_dyn_test = vector_compare(color_ptr_stat, color_ptr_dyn, "coloring ptr test");

    if(not color_ptr_stat_vs_dyn_test){
        std::cerr << "coloring ptr test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }
    all_pass = all_pass && color_ptr_stat_vs_dyn_test;

    // now we check the color sorted rows
    std::vector<int> color_sorted_rows_stat (A.get_num_rows(), 0);
    std::vector<int> color_sorted_rows_dyn (A.get_num_rows(), 0);

    int colors_sorted_dyn = 0;
    int colors_sorted_stat = 0;

    for(int color = 0; color <= max_col_dyn; color++){
        for(int row = 0; row < A.get_num_rows(); row++){
            if (colors_dynamically[row] == color){
                color_sorted_rows_dyn[colors_sorted_dyn] = row;
                colors_sorted_dyn++;
            }
            if (colors_statically[row] == color){
                color_sorted_rows_stat[colors_sorted_stat] = row;
                colors_sorted_stat++;
            }
        }
    }

    bool color_sorted_rows_stat_vs_dyn_test = vector_compare(color_sorted_rows_stat, color_sorted_rows_dyn, "coloring sorted rows test");
    
    if(not color_sorted_rows_stat_vs_dyn_test){
        std::cerr << "coloring sorted rows test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }
    all_pass = all_pass && color_sorted_rows_stat_vs_dyn_test;

    A.generate_coloring();
    // now come the comparisons with the parallel computed COR Format
    std::vector<int> parallel_color_ptr = A.get_color_pointer_vector();
    std::vector<int> parallel_color_sorted_rows = A.get_color_sorted_rows_vector();

    bool parallel_col_ptr_test = vector_compare(color_ptr_stat, parallel_color_ptr, "parallel color ptr test");
    if (not parallel_col_ptr_test){
        std::cerr << "parallel color ptr test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }

    all_pass = all_pass && parallel_col_ptr_test;

    bool parallel_col_sorted_rows_test = true;
    for(int i = 0; i <= max_col_dyn; i++){
        int begin = parallel_color_ptr[i];
        int end = parallel_color_ptr[i+1];

        for(int i = begin; i < end; i++){
            int row = parallel_color_sorted_rows[i];

            for(int j = begin; j < end; j++){
                if (color_sorted_rows_stat[j] == row){
                    break;
                }
                if (j == end - 1){
                    // in this case we did not find the row in the color_sorted_rows_stat
                    parallel_col_sorted_rows_test = false;
                }
            }
        }
    }


    if(not parallel_col_sorted_rows_test){
        std::cerr << "parallel color sorted rows test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }
    all_pass = all_pass && parallel_col_sorted_rows_test;

    // for(int i= 0; i < color_ptr_dyn.size(); i++){
    //     std::cout << "color_ptr_dyn[" << i << "] = " << color_ptr_dyn[i] << std::endl;
    // }

    // for(int i = 0; i < color_sorted_rows_dyn.size(); i++){
    //     std::cout << "color_sorted_rows_dyn[" << i << "] = " << color_sorted_rows_dyn[i] << std::endl;
    // }

    // A.print_COR_Format();
    
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
    all_pass = all_pass && coloring_test(striped_A);

    return all_pass;

}