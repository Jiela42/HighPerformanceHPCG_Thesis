#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "MatrixLib/coloring.cuh"
#include "testing.hpp"
#include <algorithm>
#include <chrono>


//////////////////////////////////////////////////////////////////////////////////////////////
// we have the single test files
//////////////////////////////////////////////////////////////////////////////////////////////

// read_save_tests
bool read_save_test(sparse_CSR_Matrix<double>& A, std::string info){
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
bool read_save_test(striped_Matrix<double>& A, std::string info){
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
// Make sure the parallel generation of the CSR Matrix is correct
bool parallel_CSR_generation_test(sparse_CSR_Matrix<double>& A_copy){

    std::string dim_info = std::to_string(A_copy.get_nx()) + "x" + std::to_string(A_copy.get_ny()) + "x" + std::to_string(A_copy.get_nz());
    
    int nx = A_copy.get_nx();
    int ny = A_copy.get_ny();
    int nz = A_copy.get_nz();

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A;
    A = problem.first;

    // copy data from GPU to CPU
    std::vector<local_int_t> row_ptr_host(nx*ny*nz + 1, 0);
    std::vector<local_int_t> col_idx_host(A_copy.get_nnz(), 0);
    std::vector<double> values_host(A_copy.get_nnz(), 0);

    if(A.get_nnz() != A_copy.get_nnz()){
        std::cerr << "nnz mismatch for generateMatrix_onGPU test for " << dim_info << std::endl;
        std::cerr << "A nnz: " << A.get_nnz() << " A_copy nnz: " << A_copy.get_nnz() << std::endl;
        return false;
    }

    CHECK_CUDA(cudaMemcpy(row_ptr_host.data(), A_copy.get_row_ptr_d(), (nx*ny*nz + 1) * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(col_idx_host.data(), A_copy.get_col_idx_d(), A_copy.get_nnz() * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(values_host.data(), A_copy.get_values_d(), A_copy.get_nnz() * sizeof(double), cudaMemcpyDeviceToHost));


    sparse_CSR_Matrix<double> A_host_copy(nx, ny, nz, A_copy.get_nnz(), MatrixType::Stencil_3D27P, values_host, row_ptr_host, col_idx_host);

    bool test_pass = A.compare_to(A_host_copy, "generateMatrix_onGPU test for " + dim_info);

    if(not test_pass){
        std::cerr << "generateMatrix_onGPU test failed for " << dim_info << std::endl;
    }

    return test_pass;
}

bool parallel_generation_from_CSR_test(sparse_CSR_Matrix<double>& A){

    // this is a redundant test.
    // we already do this in the parallel crs vs striped test.
    // but doppelt hält besser
    // first we generate the matrix on the GPU
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A_host = problem.first;

    striped_Matrix<double>* striped_A_host = A_host.get_Striped();
    std::cout << "getting striped matrix" << std::endl;
    
    // if the CSR is on the CPU, the striped matrix is generated on the CPU
    // striped_A_host.striped_Matrix_from_sparse_CSR(A_host);


    // std::cout << "striped_A_host generated" << std::endl;
    // if (striped_A_host.get_values_d() == nullptr){
        //     std::cout << "striped_A_host values_d is nullptr" << std::endl;
        // }else{
            //     std::cout << "striped_A_host values_d is not nullptr" << std::endl;
            //     std::cout << "this is the address: " << striped_A_host.get_values_d() << std::endl;
            // }
            
    A.generateMatrix_onGPU(nx, ny, nz);
    // if the CSR is on the GPU, the striped matrix is generated on the GPU
    striped_Matrix<double>* striped_A = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;
    
    // striped_A.striped_Matrix_from_sparse_CSR(A);


    // if (striped_A.get_values_d() == nullptr){
    //     std::cout << "striped_A values_d is nullptr" << std::endl;
    //     std::cout << "parallel generation test is meaningless" << std::endl;
    // }else{
    //     std::cout << "striped_A values_d is not nullptr" << std::endl;
    //     std::cout << "this is the address: " << striped_A.get_values_d() << std::endl;
    // }

    // grab the values from the GPU
    std::vector<local_int_t> j_min_i_host(striped_A->get_num_stripes(), 0);
    std::vector<double> values_host(striped_A->get_num_stripes() * nx * ny * nz, 0);

    CHECK_CUDA(cudaMemcpy(j_min_i_host.data(), striped_A->get_j_min_i_d(), striped_A->get_num_stripes() * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(values_host.data(), striped_A->get_values_d(), striped_A->get_num_stripes() * nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));

    // since we cannot just generate from j_min_i and values we do the comparison manually

    // first we check all the single values and make sure they are equal
    bool test_passed = true;

    if(striped_A->get_num_stripes() != striped_A_host->get_num_stripes()){
        std::cerr << "num stripes mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_num_rows() != striped_A_host->get_num_rows()){
        std::cerr << "num rows mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_num_cols() != striped_A_host->get_num_cols()){
        std::cerr << "num cols mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_nx() != striped_A_host->get_nx()){
        std::cerr << "nx mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_ny() != striped_A_host->get_ny()){
        std::cerr << "ny mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_nz() != striped_A_host->get_nz()){
        std::cerr << "nz mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_nnz() != striped_A_host->get_nnz()){
        std::cerr << "nnz mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_diag_index() != striped_A_host->get_diag_index()){
        std::cerr << "diag index mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    if(striped_A->get_matrix_type() != striped_A_host->get_matrix_type()){
        std::cerr << "matrix type mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }
    
    for(int i = 0; i < striped_A_host->get_num_stripes(); i++){
        if(j_min_i_host[i] != striped_A_host->get_j_min_i()[i]){
            std::cerr << "j_min_i mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
            test_passed = false;
        }
    }

    if(not vector_compare(values_host, striped_A_host->get_values())){
        std::cerr << "values mismatch for generate_striped_Matrix_from_CSR test" << std::endl;
        test_passed = false;
    }

    return test_passed;

}

//////////////////////////////////////////////////////////////////////////////////////////////
// striped vs csr tests
bool striped_csr_conversion_test_on_matrix(sparse_CSR_Matrix<double>& A){
    striped_Matrix<double>* striped_A = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;

    // striped_A.striped_Matrix_from_sparse_CSR(A);

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
    sparse_CSR_Matrix<double>* back_to_CSR = striped_A->get_CSR();
    std::cout << "getting CSR Matrix" << std::endl;
    // std::cout << "in test striped A nnz: " << striped_A.get_nnz() << std::endl;
    // back_to_CSR.sparse_CSR_Matrix_from_striped(striped_A);

    bool test_passed = A.compare_to(*back_to_CSR, "striped vs csr conversion test");

    return test_passed;
}

bool run_striped_csr_conversion_test(int nx, int ny, int nz){
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    bool all_pass = true;

    // std::cout << "normal values" << std::endl;
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

bool striped_csr_parallel_conversion_test_on_matrix(sparse_CSR_Matrix<double>&A){

    striped_Matrix<double>* striped_A = A.get_Striped();
    std::cout << "getting striped matrix" << std::endl;
    sparse_CSR_Matrix<double>* back_to_CSR = striped_A->get_CSR();
    std::cout << "getting CSR Matrix" << std::endl;
    // striped_A.striped_Matrix_from_sparse_CSR(A);
    // back_to_CSR.sparse_CSR_Matrix_from_striped(striped_A);

    // instead of copying it back by hand, we just call the function and then compare the two matrices with the built in function

    // if(A.get_coarse_Matrix() != nullptr){
    //     std::cout << "A has a coarse matrix" << std::endl;
    // } else{
    //     std::cout << "A has no coarse matrix" << std::endl;
    // }

    back_to_CSR->copy_Matrix_toCPU();

    // if(back_to_CSR.get_coarse_Matrix() != nullptr){
    //     std::cout << "back_to_CSR has a coarse matrix" << std::endl;
    // } else{
    //     std::cout << "back_to_CSR has no coarse matrix" << std::endl;
    // }

    bool test_passed = A.compare_to(*back_to_CSR, "striped vs csr parallel conversion test");

    // std::cout<< "test passed: " << test_passed << std::endl;

    return test_passed;
}

bool run_striped_csr_parallel_conversion_test(int nx, int ny, int nz){
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A = problem.first;

    bool all_pass = true;

    std::cout << "Copying to GPU on purpose" << std::endl;
    A.copy_Matrix_toGPU();

    // add some mg testing
    if(nx % 2 == 0 and ny % 2 == 0 and nz % 2 == 0 and nx / 2 > 2 and ny / 2 > 2 and nz / 2 > 2){
        A.initialize_coarse_Matrix();
    }

    // std::cout << "normal values" << std::endl;
    all_pass = all_pass && striped_csr_parallel_conversion_test_on_matrix(A);
    if (not all_pass){
        std::cerr << "striped vs csr parallel conversion test failed for normal HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    // std::cout << "random values" << std::endl;
    A.random_values(RANDOM_SEED);
    std::cout << "Copying to GPU on purpose" << std::endl;
    A.copy_Matrix_toGPU();
    all_pass = all_pass && striped_csr_parallel_conversion_test_on_matrix(A);
    if (not all_pass){
        std::cerr << "striped vs csr parallel conversion test failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }   
    // std::cout << "iterative values" << std::endl;
    A.iterative_values();
    std::cout << "Copying to GPU on purpose" << std::endl;
    A.copy_Matrix_toGPU();
    all_pass = all_pass && striped_csr_parallel_conversion_test_on_matrix(A);
    if (not all_pass){
        std::cerr << "striped vs csr parallel conversion test failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    return all_pass;
}

bool parallel_striped_generation_test(int nx, int ny, int nz){

    sparse_CSR_Matrix<double> A_csr;
    A_csr.generateMatrix_onGPU(nx, ny, nz);

    if(nx % 8 == 0 and ny % 8 == 0 and nz % 8 == 0 and nx / 8 > 2 and ny / 8 > 2 and nz / 8 > 2){
        // we check that it's devisible by 8 and still big enough to be good to do the conversions to striped with no issues
        // in this case we also initialize the MG data

        sparse_CSR_Matrix<double>* current_matrix = &A_csr;

        for(int i = 0; i < 3; i++){
            // std::cout << "doing level " << i << std::endl;
            // we do three levels of MG data
            current_matrix->initialize_coarse_Matrix();
            // std::cout << "coarse matrix initialized" << std::endl;
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    }

    striped_Matrix<double>* A = A_csr.get_Striped();

    striped_Matrix<double> A_generated_onGPU;
    A_generated_onGPU.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);


    // initialize the coarse striped matrices
    if(nx % 8 == 0 and ny % 8 == 0 and nz % 8 == 0 and nx / 8 > 2 and ny / 8 > 2 and nz / 8 > 2){

        striped_Matrix<double>* current_Matrix = &A_generated_onGPU;

        for(int i = 0; i < 3; i++){
            // we do three levels of MG data
            current_Matrix->initialize_coarse_Matrix();
            current_Matrix = current_Matrix->get_coarse_Matrix();
        }

    }

    // compare the two
    bool test_pass = A->compare_to(A_generated_onGPU);

    if (not test_pass){
        std::cerr << "parallel striped generation test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    return test_pass;
}

bool coloring_test(striped_Matrix<double>& A){

    bool all_pass = true;
    // std::cout << "coloring test" << std::endl;

    // first we get the long hand computed colors
    std::vector<local_int_t> colors_dynamically = color_for_forward_pass(A);

    // std::cout << "dynamic colors done" << std::endl;

    // then we get the colors from the matrix
    std::vector<local_int_t> colors_statically (A.get_num_rows(), 0);

    for (local_int_t i = 0 ; i < A.get_num_rows(); i++){
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

    local_int_t max_col_dyn = *std::max_element(colors_dynamically.begin(), colors_dynamically.end());
    local_int_t max_col_stat = *std::max_element(colors_statically.begin(), colors_statically.end());

    // std::cout << "num colors " << max_col_dyn + 1 << std::endl;

    // std::cout << "size: " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;


    std::vector<local_int_t> color_ptr_dyn(max_col_dyn+2, 0);
    std::vector<local_int_t> color_ptr_stat(max_col_stat+2, 0);

    for(local_int_t i = 0; i <= max_col_dyn; i++){
        local_int_t color_ct_i = count(colors_dynamically.begin(), colors_dynamically.end(), i);
        color_ptr_dyn[i+1] = color_ptr_dyn[i] + count(colors_dynamically.begin(), colors_dynamically.end(), i);
        color_ptr_stat[i+1] = color_ptr_dyn[i] + std::count(colors_statically.begin(), colors_statically.end(), i);
        // std::cout << "color " << i << " has " << color_ct_i << " rows" << std::endl;
    }

    bool color_ptr_stat_vs_dyn_test = vector_compare(color_ptr_stat, color_ptr_dyn, "coloring ptr test");

    if(not color_ptr_stat_vs_dyn_test){
        std::cerr << "coloring ptr test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }
    all_pass = all_pass && color_ptr_stat_vs_dyn_test;

    // now we check the color sorted rows
    std::vector<local_int_t> color_sorted_rows_stat (A.get_num_rows(), 0);
    std::vector<local_int_t> color_sorted_rows_dyn (A.get_num_rows(), 0);

    int colors_sorted_dyn = 0;
    int colors_sorted_stat = 0;

    for(local_int_t color = 0; color <= max_col_dyn; color++){
        for(local_int_t row = 0; row < A.get_num_rows(); row++){
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

    A.generate_coloring(true);
    // now come the comparisons with the parallel computed COR Format
    std::vector<local_int_t> parallel_color_ptr = A.get_color_pointer_vector();
    std::vector<local_int_t> parallel_color_sorted_rows = A.get_color_sorted_rows_vector();

    bool parallel_col_ptr_test = vector_compare(color_ptr_stat, parallel_color_ptr, "parallel color ptr test");
    if (not parallel_col_ptr_test){
        std::cerr << "parallel color ptr test failed for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    }

    all_pass = all_pass && parallel_col_ptr_test;

    bool parallel_col_sorted_rows_test = true;
    for(local_int_t i = 0; i <= max_col_dyn; i++){
        local_int_t begin = parallel_color_ptr[i];
        local_int_t end = parallel_color_ptr[i+1];

        for(local_int_t i = begin; i < end; i++){
            local_int_t row = parallel_color_sorted_rows[i];

            for(local_int_t j = begin; j < end; j++){
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

bool box_coloring_test(striped_Matrix<double>& A){

    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    int bx = 3;
    int by = 3;
    int bz = 3;
    int num_colors = bx * by * bz;

    // first color the matrix
    A.generate_box_coloring(bx, by, bz);

    // now we get the colors from the matrix
    std::vector<local_int_t> color_idx_ptr(num_colors + 1, 0);
    std::vector<local_int_t> color_sorted_rows(A.get_num_rows(), 0);

    CHECK_CUDA(cudaMemcpy(color_idx_ptr.data(), A.get_color_pointer_d(false), 28 * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(color_sorted_rows.data(), A.get_color_sorted_rows_d(false), A.get_num_rows() * sizeof(local_int_t), cudaMemcpyDeviceToHost));

    for(int i = 1; i <= num_colors; i ++){
        int color = i-1;
        int start = color_idx_ptr[color];
        int end = color_idx_ptr[color+1];

        int num_color_cols = nx / bx;
        int num_color_rows = ny / by;
        int num_color_faces = nz / bz;
        
        int color_offs_x = color % bx;
        int color_offs_y = (color - color_offs_x) % (bx * by) / bx;
        int color_offs_z = (color - color_offs_x - bx * color_offs_y) / (bx * by);
        
        num_color_cols = (color_offs_x < nx % bx) ? (num_color_cols + 1) : num_color_cols;
        num_color_rows = (color_offs_y < ny % by) ? (num_color_rows + 1) : num_color_rows;
        num_color_faces = (color_offs_z < nz % bz) ? (num_color_faces + 1) : num_color_faces;
        
        int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;
        std::vector<int> symGS_color_i(num_nodes_with_color);
        
        assert(num_nodes_with_color == end-start);
        // std::cout << "nodes with color: " << color << ": " << num_nodes_with_color << std::endl;

        // populate the symGS_color_i
        for(int r = 0; r < num_nodes_with_color; r++){
            // find out the position of the node (only considering faces, cols and rows that actually have that color)
            int ix = r % num_color_cols;
            int iy = (r % (num_color_cols * num_color_rows)) / num_color_cols;
            int iz = r / (num_color_cols * num_color_rows);
            
            // adjust the counter to the correct position when all nodes are considered
            ix = ix * bx + color_offs_x;
            iy = iy * by + color_offs_y;
            iz = iz * bz + color_offs_z;

            int row = ix + iy * nx + iz * nx * ny;
            symGS_color_i[r] = row;
        }

        int prev_row = -1;
        // now we check that every row in the COR format is also calculated by symGS
        for(int row_idx = start; row_idx < end; row_idx++){
            int row = color_sorted_rows[row_idx];
            // make sure the rows increase for each color
            assert(row > prev_row);
            prev_row = row;
            bool found_row = false;
            for(int j = 0; j < num_nodes_with_color; j++){
                if(symGS_color_i[j] == row){
                    found_row = true;
                    break;
                }
            }
            if (not found_row){
                std::cout << "start: " << start << " end: " << end << std::endl;
                std::cout << "row " << row << " not found in symGS_color_i, color: " << color << std::endl;
                std::cout << "symGS_color_i: ";
                for(int j = 0; j < num_nodes_with_color; j++){
                    std::cout << symGS_color_i[j] << " ";
                }
                std::cout << std::endl;
                std::cout << "color_sorted_rows: ";
                for(int j = start; j < end; j++){
                    std::cout << color_sorted_rows[j] << " ";
                }
                std::cout << std::endl;
            }
            assert (found_row);
        }

    }
    return true;


}


bool run_MG_data_tests(sparse_CSR_Matrix<double>& A){

    bool c2f_test = true;

    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();



    sparse_CSR_Matrix<double>* current_matrix = &A;

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> host_matrix = problem.first;

    while(current_matrix->get_coarse_Matrix() != nullptr){

        local_int_t coarse_n_rows = current_matrix->get_coarse_Matrix()->get_num_rows();
        local_int_t fine_n_rows = current_matrix->get_num_rows();
        // A is on the GPU and should have an c2f operator

        current_matrix = current_matrix->get_coarse_Matrix();

        host_matrix.initialize_coarse_Matrix();
        host_matrix = *(host_matrix.get_coarse_Matrix());

        std::vector<local_int_t> c2f_device(fine_n_rows, 0);

        CHECK_CUDA(cudaMemcpy(c2f_device.data(), current_matrix->get_f2c_op_d(), fine_n_rows * sizeof(int), cudaMemcpyDeviceToHost));


        std::vector<local_int_t> c2f_host = host_matrix.get_f2c_op();

        // check if any of the c2f values are negative
        for(local_int_t i = 0; i < fine_n_rows; i++){
            if(c2f_device[i] < 0){
                std::cerr << "c2f device has negative value at index " << i << std::endl;
                c2f_test = false;
            }
        }

        // std::cout << "device size: " << c2f_device.size() << " host size: " << c2f_host.size() << std::endl;
        // std::cout << "device[1]: " << c2f_device[1] << " host[1]: " << c2f_host[1] << std::endl;
        // std::cout << "device[2]: " << c2f_device[2] << " host[2]: " << c2f_host[2] << std::endl;
        // std::cout << "device[3]: " << c2f_device[3] << " host[3]: " << c2f_host[3] << std::endl;


        c2f_test = c2f_test && vector_compare(c2f_host, c2f_device, "c2f test");

        if(not c2f_test){
            std::cerr << "c2f test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
        }
    }

    return c2f_test;


}
//////////////////////////////////////////////////////////////////////////////////////////////
// for each size we recieve we generate the matrices and run all the tests

bool run_all_matrixLib_tests(int nx, int ny, int nz){
    
    std::string dim_info = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);

    // run tests on standard 3d27p matrices
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    striped_Matrix<double>* striped_A = A.get_Striped();
    // std::cout << "getting striped matrix" << std::endl;
    // striped_A.striped_Matrix_from_sparse_CSR(A);

    bool all_pass = true;

    // std::cout << "nx % 8: " << nx % 8 << " ny % 8: " << ny % 8 << " nz % 8: " << nz % 8 << std::endl;
    // std::cout << "nx / 8: " << nx / 8 << " ny / 8: " << ny / 8 << " nz / 8: " << nz / 8 << std::endl;
    if(nx % 8 == 0 and ny % 8 == 0 and nz % 8 == 0 and nx / 8 > 2 and ny / 8 > 2 and nz / 8 > 2){
        // we check that it's devisible by 8 and still big enough to be good to do the conversions to striped with no issues
        // in this case we also initialize the MG data

        sparse_CSR_Matrix<double>* current_matrix = &A;

        for(int i = 0; i < 3; i++){
            // std::cout << "doing level " << i << std::endl;
            // we do three levels of MG data
            current_matrix->initialize_coarse_Matrix();
            // std::cout << "coarse matrix initialized" << std::endl;
            current_matrix = current_matrix->get_coarse_Matrix();
        }

    }

    // all_pass = all_pass && read_save_test(A, "read_save_test on 3d27p CSR Matrix for " + dim_info);
    // read_save_test(striped_A, "read_save_test on 3d27p striped Matrix for " + dim_info);
    
    std::cout << "testing " << dim_info << std::endl;

    all_pass = all_pass && parallel_CSR_generation_test(A);
    // std::cout << "parallel CSR generation test passed for " << dim_info << std::endl;
    all_pass = all_pass && run_striped_csr_conversion_test(nx, ny, nz);
    // std::cout << "striped vs csr sequential conversion test passed for " << dim_info << std::endl;
    all_pass = all_pass && run_striped_csr_parallel_conversion_test(nx, ny, nz);
    // std::cout << "striped vs csr parallel conversion test passed for " << dim_info << std::endl;
    all_pass = all_pass && parallel_generation_from_CSR_test(A);
    all_pass = all_pass && parallel_striped_generation_test(nx, ny, nz);
    all_pass = all_pass && coloring_test(*striped_A);
    all_pass = all_pass && box_coloring_test(*striped_A);
    // std::cout << "coloring test passed for " << dim_info << std::endl;
    all_pass = all_pass && run_MG_data_tests(A);


    return all_pass;
}