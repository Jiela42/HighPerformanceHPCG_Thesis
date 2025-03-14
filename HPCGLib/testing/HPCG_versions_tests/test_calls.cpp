#include "testing.hpp"

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveStriped.cuh"

#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>


// these functions are really just wrappers to check for correctness
// hence the only thing the function needs to allcoate is space for the outputs
// depending on the versions they may require different inputs, hence the method overloading

// file based tests

bool test_CG(
    HPCG_functions<double>& implementation)
{

    if (implementation.norm_based){
        std::cerr << "CG norm based tests are not supported" << std::endl;
        return true;
    }


    bool all_pass = true;
    std::string test_folder = HPCG_OUTPUT_TEST_FOLDER;

    // the file_based tests require a tolerance of zero and a max iteration of 50
    
    double original_tolerance = implementation.get_CGTolerance();
    int original_max_iters = implementation.get_maxCGIters();

    implementation.set_CGTolerance(0.0);
    implementation.set_maxCGIters(50);

    // iterate through the test files
    for (const auto& entry : std::filesystem::directory_iterator(test_folder)) {
        std::string test_file_path = entry.path().string();
        std::string test_folder_name = entry.path().filename().string();

        std::ifstream file(test_file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << test_file_path << std::endl;
            all_pass = false;
            continue;
        }

        // std::cout << "Running test: " << test_file_path << std::endl;
        // std::cout << "Folder name: " << test_folder_name << std::endl;

        size_t underscore_pos = test_folder_name.find('_');
        std::string method_name = (underscore_pos != std::string::npos) ? test_folder_name.substr(0, underscore_pos) : test_folder_name;

        // std::cout << "Method name: " << method_name << std::endl;


        if(method_name == "CG"){ // and num_tests < 1){
            // num_tests++;

            size_t first_underscore_pos = test_folder_name.find('_');
            size_t second_underscore_pos = test_folder_name.find('_', first_underscore_pos + 1);
            std::string preconditioning_info = (first_underscore_pos != std::string::npos && second_underscore_pos != std::string::npos) ? test_folder_name.substr(first_underscore_pos + 1, second_underscore_pos - first_underscore_pos - 1) : "";

            // std::cout << "Preconditioning info: " << preconditioning_info << std::endl;
            // std::cout << "num_tests: " << num_tests << std::endl;


            std::string dimA_file_path = test_file_path + "/dimA.txt";
            std::string b_file_path = test_file_path + "/b.txt";
            std::string x_beforeCG_file_path = test_file_path + "/x_beforeCG.txt";
            std::string x_afterCG_file_path = test_file_path + "/x_afterCG.txt";

            int nx; int ny; int nz; int num_rows; int num_cols;

            // open and read the dimensions from dimA.txt
            std::ifstream dimA_file(dimA_file_path);
            if (!dimA_file.is_open()) {
                std::cerr << "Failed to open file: " << dimA_file_path << std::endl;
                all_pass = false;
                continue;
            }

            std::string line;
            while (std::getline(dimA_file, line)) {
                if (line.find("Number of Rows:") != std::string::npos) {
                    num_rows = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("Number of Columns:") != std::string::npos) {
                    num_cols = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("nx:") != std::string::npos) {
                    nx = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("ny:") != std::string::npos) {
                    ny = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("nz:") != std::string::npos) {
                    nz = std::stoi(line.substr(line.find(":") + 1));
                }
            }
            dimA_file.close();

            std::vector<double> b_host(num_rows, 0.0);
            std::vector<double> x_beforeCG_host(num_rows, 0.0);
            std::vector<double> x_afterCG_host(num_rows, 0.0);

            // now read the vectors
            std::ifstream b_file(b_file_path);
            if (b_file.is_open()) {
                for (int i = 0; i < num_rows && b_file >> b_host[i]; ++i);
                b_file.close();
            } else {
                std::cerr << "Failed to open file: " << b_file_path << std::endl;
                all_pass = false;
                continue;
            }

            std::ifstream x_beforeCG_file(x_beforeCG_file_path);
            if (x_beforeCG_file.is_open()) {
                for (int i = 0; i < num_rows && x_beforeCG_file >> x_beforeCG_host[i]; ++i);
                x_beforeCG_file.close();
            } else {
                std::cerr << "Failed to open file: " << x_beforeCG_file_path << std::endl;
                all_pass = false;
                continue;
            }

            std::ifstream x_afterCG_file(x_afterCG_file_path);
            if (x_afterCG_file.is_open()) {
                for (int i = 0; i < num_rows && x_afterCG_file >> x_afterCG_host[i]; ++i);
                x_afterCG_file.close();
            } else {
                std::cerr << "Failed to open file: " << x_afterCG_file_path << std::endl;
                all_pass = false;
                continue;
            }

            double * b_d;
            double * x_d;

            CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&x_d, num_rows * sizeof(double)));

            CHECK_CUDA(cudaMemcpy(b_d, b_host.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(x_d, x_beforeCG_host.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

            // // print the first 5 elements of x
            // for (int i = 0; i < 5; i++) {
            //     std::cout << x_beforeCG_host[i] << " ";
            // }
            // std::cout << std::endl;

            // print the size
            // std::cout << "Size: " << nx << "x" << ny << "x" << nz << std::endl;

            // when we are testing the size 128x128x128 causes out of memory so we skip that test
            if(nx == 128 and ny == 128 and nz == 128){
                std::cerr << "Skipping test for size 128x128x128 to avoid out of memory during benchmark tests" << std::endl;
                continue;
            }

            // make A
            sparse_CSR_Matrix<double> A;
            A.generateMatrix_onGPU(nx, ny, nz);

            // preallocate an int and two doubles for the output
            int n_iters;
            double normr;
            double normr0;

            if(preconditioning_info == "noPreconditioning"){

                implementation.doPreconditioning = false;
                if (implementation.implementation_type == Implementation_Type::STRIPED){
                    striped_Matrix<double>* A_striped = A.get_Striped();

                    // we might need a coloring precomputed for some SymGS implementations
                    A_striped->generate_coloring();

                    implementation.compute_CG(*A_striped, b_d, x_d, n_iters, normr, normr0);
                    std::cout << "CG took " << n_iters << " iterations for size " << nx << "x" << ny << "x" << nz << " without Preconditioning, with a normr/normr0 of " << normr/normr0 << std::endl;

                } else{
                    std::cout << "CG not implemented for this implementation" << std::endl;
                    all_pass = false;
                }

            } else if (nx >= 24 and ny >= 24 and nz >= 24 and VERIFY_CG_WITH_PRECONDITIONING){
                // when we do preconditioning we need at least 3 layers of coarse matrices to be able to compare to an HPCG file and we cannot do 2x2x2 matrices
                
                // initialize the coarse matrices to be used when preconditioning
                sparse_CSR_Matrix<double>* current_matrix = &A;
    
                for(int i = 0; i < 3; i++){
                    current_matrix->initialize_coarse_Matrix();
                    current_matrix = current_matrix->get_coarse_Matrix();
                }

                
                implementation.doPreconditioning = true;
                
                if (implementation.implementation_type == Implementation_Type::STRIPED){
                    striped_Matrix<double>* A_striped = A.get_Striped();
                    
                    // we might need a coloring precomputed
                    A_striped->generate_coloring();

                    implementation.compute_CG(*A_striped, b_d, x_d, n_iters, normr, normr0);

                    std::cout << "CG took " << n_iters << " iterations for size " << nx << "x" << ny << "x" << nz << " with Preconditioning, with a normr/normr0 of " << normr/normr0 << std::endl;
                } else{
                    std::cout << "CG not implemented for this implementation" << std::endl;
                    all_pass = false;
                }
    
            } else{
                if(not VERIFY_CG_WITH_PRECONDITIONING){
                    std::cout << "Skipping preconditioning tests in the interest of time" << std::endl;
                    continue;
                }
                // if we would need to do preconditioning but the matrix is too small we move on
                std::cerr << "Matrix too small for preconditioning" << std::endl;
                continue;
            }

    
            // now get the result and compare
            std::vector<double> computed_result(num_rows, 0.0);
            CHECK_CUDA(cudaMemcpy(computed_result.data(), x_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

            // compare the results
            // bool test_pass = vector_compare(x_afterCG_host, computed_result);

            // if(not test_pass){
            //     std::cerr << "CG test failed for size " << nx << "x" << ny << "x" << nz << " for implementation: " << implementation.version_name << std::endl;
            //     all_pass = false;
            //     return all_pass;
            // }

            
            // else {
            //     // print the first 5 elements of the vectors
            //     for (int i = 0; i < 5; i++) {
            //         std::cout << computed_result[i] << " " << x_afterCG_host[i] << std::endl;
            //     }
            // }
        }
        file.close();
    }

    // restore the original values
    implementation.set_CGTolerance(original_tolerance);
    implementation.set_maxCGIters(original_max_iters); 

    return all_pass;
}

bool test_MG(
    HPCG_functions<double>& implementation)
{

    if(implementation.norm_based){
        std::cerr << "MG norm based tests are not supported" << std::endl;
        return true;
    }

    bool all_pass = true;
    std::string test_folder = HPCG_OUTPUT_TEST_FOLDER;

    // iterate through the test files
    for (const auto& entry : std::filesystem::directory_iterator(test_folder)) {
        std::string test_file_path = entry.path().string();
        std::string test_folder_name = entry.path().filename().string();

        std::ifstream file(test_file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << test_file_path << std::endl;
            all_pass = false;
            continue;
        }

        // std::cout << "Running test: " << test_file_path << std::endl;
        // std::cout << "Folder name: " << test_folder_name << std::endl;

        size_t underscore_pos = test_folder_name.find('_');
        std::string method_name = (underscore_pos != std::string::npos) ? test_folder_name.substr(0, underscore_pos) : test_folder_name;

        // std::cout << "Method name: " << method_name << std::endl;

        if(method_name == "MG"){
            std::string dimA_file_path = test_file_path + "/dimA.txt";
            std::string b_computed_file_path = test_file_path + "/b_computed.txt";
            std::string x_overlap_file_path = test_file_path + "/x_overlap.txt";
            std::string x_overlap_after_mg_file_path = test_file_path + "/x_overlap_after_mg.txt";

            int nx; int ny; int nz; int num_rows; int num_cols;

            // open and read the dimensions from dimA.txt
            std::ifstream dimA_file(dimA_file_path);
            if (!dimA_file.is_open()) {
                std::cerr << "Failed to open file: " << dimA_file_path << std::endl;
                all_pass = false;
                continue;
            }

            std::string line;
            while (std::getline(dimA_file, line)) {
                if (line.find("Number of Rows:") != std::string::npos) {
                    num_rows = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("Number of Columns:") != std::string::npos) {
                    num_cols = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("nx:") != std::string::npos) {
                    nx = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("ny:") != std::string::npos) {
                    ny = std::stoi(line.substr(line.find(":") + 1));
                } else if (line.find("nz:") != std::string::npos) {
                    nz = std::stoi(line.substr(line.find(":") + 1));
                }
            }
            dimA_file.close();
            // if(not (nx == 64 and ny ==128 and nz == 32))
            // {continue;}

            // because of the MG data the following sizes cause out of memory for benchmark tests
            if((nx == 128 and ny == 128 and nz == 128) or
                (nx == 64 and ny == 64 and nz == 64) or
                (nx == 32 and ny == 64 and nz == 128) or
                (nx == 64 and ny == 32 and nz == 128) or
                (nx == 32 and ny == 128 and nz == 64) or
                (nx == 128 and ny == 32 and nz == 64)){
                std::cerr << "Skipping test for size " << nx << "x" << ny << "x" << nz << " to avoid out of memory during benchmark tests" << std::endl;
                continue;
            }

            if(nx >= 24 and ny >= 24 and nz >= 24){
                // we need at 3 layers of coarse matrices to be able to compare to an HPCG file and we cannot do 2x2x2 matrices

                // allocate the memory for the host vectors
                // std::cout << "nx: " << nx << " ny: " << ny << " nz: " << nz << std::endl;
                // std::cout << "Number of Rows: " << num_rows << std::endl;
                std::vector<double> b_computed_host(num_rows, 0.0);
                std::vector<double> x_overlap_host(num_rows, 0.0);
                std::vector<double> x_overlap_after_mg_host(num_rows, 0.0);
    
                // now read the vectors
                std::ifstream b_computed_file(b_computed_file_path);
                if (b_computed_file.is_open()) {
                    for (int i = 0; i < num_rows && b_computed_file >> b_computed_host[i]; ++i);
                    b_computed_file.close();
                } else {
                    std::cerr << "Failed to open file: " << b_computed_file_path << std::endl;
                    all_pass = false;
                    continue;
                }
    
                std::ifstream x_overlap_file(x_overlap_file_path);
                if (x_overlap_file.is_open()) {
                    for (int i = 0; i < num_rows && x_overlap_file >> x_overlap_host[i]; ++i);
                    x_overlap_file.close();
                } else {
                    std::cerr << "Failed to open file: " << x_overlap_file_path << std::endl;
                    all_pass = false;
                    continue;
                }
    
                std::ifstream x_overlap_after_mg_file(x_overlap_after_mg_file_path);
                if (x_overlap_after_mg_file.is_open()) {
                    for (int i = 0; i < num_rows && x_overlap_after_mg_file >> x_overlap_after_mg_host[i]; ++i);
                    x_overlap_after_mg_file.close();
                } else {
                    std::cerr << "Failed to open file: " << x_overlap_after_mg_file_path << std::endl;
                    all_pass = false;
                    continue;
                }
    
                // // print the first 5 elements of the vectors
                // std::cout << "b_computed: ";
                // for (int i = 0; i < 5; i++) {
                //     std::cout << b_computed_host[i] << " ";
                // }
                // std::cout << std::endl;
                // std::cout << "x_overlap: ";
                // for (int i = 0; i < 5; i++) {
                //     std::cout << x_overlap_host[i] << " ";
                // }
                // std::cout << std::endl;
                // std::cout << "x_overlap_after_mg: ";
                // for (int i = 0; i < 5; i++) {
                //     std::cout << x_overlap_after_mg_host[i] << " ";
                // }
                // std::cout << std::endl;
    
                // allocate the memory for the device vectors
                double * b_computed_d;
                double * x_overlap_d;
    
                CHECK_CUDA(cudaMalloc(&b_computed_d, num_rows * sizeof(double)));
                CHECK_CUDA(cudaMalloc(&x_overlap_d, num_rows * sizeof(double)));
    
                CHECK_CUDA(cudaMemcpy(b_computed_d, b_computed_host.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(x_overlap_d, x_overlap_host.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    
                // make A
                sparse_CSR_Matrix<double> A;
                A.generateMatrix_onGPU(nx, ny, nz);
    
                // also initialize the MG data
                sparse_CSR_Matrix<double>* current_matrix = &A;
    
                for(int i = 0; i < 3; i++){
                    current_matrix->initialize_coarse_Matrix();
                    current_matrix = current_matrix->get_coarse_Matrix();
                }
    
                // if(implementation.implementation_type == Implementation_Type::CSR){
                    
                //     implementation.compute_MG(A, b_computed_d, x_overlap_d);
    
                // } else

                if (implementation.implementation_type == Implementation_Type::STRIPED){

                    striped_Matrix<double>* A_striped = A.get_Striped();

                    // we might need a coloring precomputed
                    A_striped->generate_coloring();
    
                    // test the MG function
                    implementation.compute_MG(*A_striped, b_computed_d, x_overlap_d);
                } else{
                    std::cout << "MG not implemented for this implementation" << std::endl;
                    all_pass = false;
                }
    
                // now get the result and compare
                std::vector<double> computed_result(num_rows, 0.0);
                CHECK_CUDA(cudaMemcpy(computed_result.data(), x_overlap_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    
                // compare the results
                bool test_pass = vector_compare(x_overlap_after_mg_host, computed_result);
    
                if(not test_pass){
                    std::cerr << "MG test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
                    all_pass = false;
                }
                // else {
                //     // print the first 5 elements of the vectors
                //     for (int i = 0; i < 5; i++) {
                //         std::cout << computed_result[i] << " " << x_overlap_after_mg_host[i] << std::endl;
                //         std::cout << (abs(computed_result[i] - x_overlap_after_mg_host[i]) < error_tolerance) << std::endl;
                //     }
                // }
            }
        }

        file.close();
    }
    // std::cout << "MG tested for implementation: " << implementation.version_name << std::endl;
    return all_pass;
}

// in this case both versions require the same inputs 
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A,
    double * x_d // the vectors x is already on the device
        
){  

    int num_rows = A.get_num_rows();
    std::vector<double> y_baseline(num_rows, 0.0);
    std::vector<double> y_uut(num_rows, 0.0);

    double * y_baseline_d;
    double * y_uut_d;

    CHECK_CUDA(cudaMalloc(&y_baseline_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_uut_d, num_rows * sizeof(double)));

    baseline.compute_SPMV(A,
                          x_d, y_baseline_d);

    uut.compute_SPMV(A,
                    x_d, y_uut_d);

    // and now we need to copy the result back and de-allocate the memory
    CHECK_CUDA(cudaMemcpy(y_baseline.data(), y_baseline_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y_uut.data(), y_uut_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(y_baseline_d));
    CHECK_CUDA(cudaFree(y_uut_d));

    bool test_pass = vector_compare(y_baseline, y_uut);
    return test_pass;
}

// in this case the baseline requires CSR and the UUT requires both CSR and striped
bool test_SPMV(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & striped_A, // we pass A for the metadata
    double * x_d // the vectors x is already on the device
        
){
    int num_rows = striped_A.get_num_rows();
    
    sparse_CSR_Matrix<double> * A = striped_A.get_CSR();
    // A.sparse_CSR_Matrix_from_striped(striped_A);

    int num_rows_baseline = A->get_num_rows();
    std::vector<double> y_baseline(num_rows, 0.0);
    std::vector<double> y_uut(num_rows, 0.0);

    double * y_baseline_d;
    double * y_uut_d;

    CHECK_CUDA(cudaMalloc(&y_baseline_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_uut_d, num_rows * sizeof(double)));

    baseline.compute_SPMV(*A,
                          x_d, y_baseline_d);

    uut.compute_SPMV(striped_A,
                    x_d, y_uut_d);
    
    // and now we need to copy the result back and de-allocate the memory
    CHECK_CUDA(cudaMemcpy(y_baseline.data(), y_baseline_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y_uut.data(), y_uut_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(y_baseline_d));
    CHECK_CUDA(cudaFree(y_uut_d));

    // compare the results

    // for my sanity print the first 10 elements

    // std::cout << "SPMV results" << std::endl;

    // for(int i = 0; i < 10; i++){
    //     std::cout << "baseline[" << i << "] = " << y_baseline[i] << " uut[" << i << "] = " << y_uut[i] << std::endl;
    // }

    bool test_pass = vector_compare(y_baseline, y_uut);

    return test_pass;
}

bool test_Dot(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & A, // we pass A for the metadata
    double * x_d, double * y_d // the vectors x, y and result are already on the device
    ){

    // sparse_CSR_Matrix<double> A_CSR;
    // A_CSR.sparse_CSR_Matrix_from_striped(A);

    double result_baseline = 0.0;
    double result_uut = 0.0;

    // allocate the memory for the result
    double * result_baseline_d;
    double * result_uut_d;

    CHECK_CUDA(cudaMalloc(&result_baseline_d, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_uut_d, sizeof(double)));

    baseline.compute_Dot(A, x_d, y_d, result_baseline_d);
    uut.compute_Dot(A, x_d, y_d, result_uut_d);

    // and now we need to copy the result back and de-allocate the memory
    CHECK_CUDA(cudaMemcpy(&result_baseline, result_baseline_d, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&result_uut, result_uut_d, sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(result_baseline_d));
    CHECK_CUDA(cudaFree(result_uut_d));

    // and now we need to copy the result back and de-allocate the memory
    bool test_pass = double_compare(result_baseline, result_uut);

    if (not test_pass){
        std::cout << "Dot product failed for implementation: "<< uut.version_name << " baseline = " << result_baseline << " uut = " << result_uut << std::endl;
    }

    return test_pass;
}

bool test_Dot(
    HPCG_functions<double>& uut,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d // the vectors x, y and result are already on the device
    ){

        // get the result on the device
        double * result_uut_d;
        double result_uut = 0.0;

        CHECK_CUDA(cudaMalloc(&result_uut_d, sizeof(double)));

        uut.compute_Dot(A, x_d, y_d, result_uut_d);

        // and now we need to copy the result back and de-allocate the memory
        CHECK_CUDA(cudaMemcpy(&result_uut, result_uut_d, sizeof(double), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(result_uut_d));

        double result_baseline = 0.0;

        // calculate the baseline result (on the host like a mooron)
        std::vector <double> x_h(A.get_num_rows());
        std::vector <double> y_h(A.get_num_rows());

        CHECK_CUDA(cudaMemcpy(x_h.data(), x_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(y_h.data(), y_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));

        for(int i = 0; i < A.get_num_rows(); i++){
            result_baseline += x_h[i] * y_h[i];
        }

        // and now we need to copy the result back and de-allocate the memory
        bool test_pass = double_compare(result_baseline, result_uut);

        if(not test_pass){
            std::cout << "Dot product failed for Implementation: " << uut.version_name << " baseline = " << result_baseline << " uut = " << result_uut << std::endl;
        }

        return test_pass;

    }

// this is a minitest, it can be called to do some rudimentary testing (currently only for striped Matrices)
bool test_Dot(
    HPCG_functions<double>& uut,
    int nx, int ny, int nz
){

    // make a matrix (for some reason we need it) (num_rows is the only thing we need to get from the matrix)
    striped_Matrix<double> A_striped;
    A_striped.set_num_rows(nx * ny * nz);
    

    // create two vectors
    std::vector<double> x(nx * ny * nz, 2.0);
    std::vector<double> y(nx * ny * nz, 0.5);
    // std::cout << "we use this function" << std::endl;

    double result = 0.0;

    srand(RANDOM_SEED);

    for(int i = 0; i < nx * ny * nz; i++){
        double a = (double)rand() / RAND_MAX;
        double b = (double)rand() / RAND_MAX;
        x[i] = a;
        y[i] = b;
        result += a * b;
        // result += x[i] * y[i];
    }

    // allocate x and y on the device
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, nx * ny * nz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, nx * ny * nz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));

    uut.compute_Dot(A_striped, x_d, y_d, result_d);

    // get result back
    double result_uut = 42;
    CHECK_CUDA(cudaMemcpy(&result_uut, result_d, sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(result_d));

    bool test_pass = relaxed_double_compare(result, result_uut, 1e-10);
    if (not test_pass){
        std::cout << "Dot product failed for implementation: " << uut.version_name << " baseline = " << result << " uut = " << result_uut << std::endl;
    }
    // else {
    //     std::cout << "Dot product passed: baseline = " << result << " uut = " << result_uut << std::endl;
    // }

    return test_pass;
}

bool test_WAXPBY(
    HPCG_functions<double>& uut,
    striped_Matrix<double> & A,
    double * x_d, double * y_d
){
    // this one runs a bunch of tests for the WAXPBY function

    srand(RANDOM_SEED);

    double a = (double)rand() / RAND_MAX;
    double b = (double)rand() / RAND_MAX;

    
    bool all_pass = test_WAXPBY(uut, A, x_d, y_d, 0.0, 0.0);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, 0.0, 1.0);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, 1.0, 0.0);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, 1.0, 1.0);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, a, 1.0);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, 1.0, b);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, a, 0.0);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, 0.0, b);
    all_pass = all_pass && test_WAXPBY(uut, A, x_d, y_d, a, b);

    return all_pass;

}

bool test_WAXPBY(
    HPCG_functions<double>& uut,
    striped_Matrix<double> & A,
    double * x_d, double * y_d,
    double alpha, double beta
)
    
    {

        // x & y should be random vectors, so let's quickly grab them
        std::vector<double> x_host(A.get_num_rows());
        std::vector<double> y_host(A.get_num_rows());
        std::vector<double> w_host(A.get_num_rows());
        std::vector<double> w_baseline(A.get_num_rows());


        // allocate result vector on device
        double * w_d;
        CHECK_CUDA(cudaMalloc(&w_d, A.get_num_rows() * sizeof(double)));

        // grab the vectors from the device
        CHECK_CUDA(cudaMemcpy(x_host.data(), x_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(y_host.data(), y_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));

        uut.compute_WAXPBY(A, x_d, y_d, w_d, alpha, beta);

        CHECK_CUDA(cudaMemcpy(w_baseline.data(), w_d, A.get_num_rows() * sizeof(double), cudaMemcpyDeviceToHost));

        for(int i = 0; i < A.get_num_rows(); i++){
            w_host[i] = alpha * x_host[i] + beta * y_host[i];
        }

        bool test_pass = vector_compare(w_host, w_baseline);
        
        if (not test_pass){
            std::cout << "WAXPBY test failed for implementation: " << uut.version_name << std::endl;
        }

        return test_pass;
    }

bool test_SymGS(
    HPCG_functions<double>&uut,
    sparse_CSR_Matrix<double> & A
    ){
    // This is the mini test for the SymGS function

    // std::cout << "SymGS Mini test" << std::endl;

    std::vector<std::vector<double>> A_dense = {
        {1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.},
        {1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0.},
        {0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.},
        {0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.},
        {1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.},
        {1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0.},
        {0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.},
        {0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1.},
        {1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0.},
        {1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0.},
        {0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1.},
        {0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1.},
        {1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0.},
        {1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0.},
        {0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1.},
        {0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.}
    };

    // Define the vector y
    std::vector<double> y = {8., 9., 9., 8., 8., 9., 9., 8., 8., 9., 9., 8., 8., 9., 9., 8.};

    // Define the solution vector
    std::vector<double> solution = {15., -7., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

    sparse_CSR_Matrix<double> A_csr (A_dense);
    int nnz = A_csr.get_nnz();
    int num_rows = A_csr.get_num_rows();

    // put A onto gpu
    std::cout << "Copying A to GPU, you can safely ignore the following warning." << std::endl;
    A_csr.copy_Matrix_toGPU();

    // A_csr.print();

    // Allocate the memory on the device
    double * y_d;
    double * x_d;

    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_rows * sizeof(double)));

    // Copy the data to the device
    CHECK_CUDA(cudaMemset(x_d, 0.0, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    // run the symGS function
    uut.compute_SymGS(A_csr, x_d, y_d);

    // get the result back
    std::vector<double> x(num_rows, 0.0);

    CHECK_CUDA(cudaMemcpy(x.data(), x_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    // free the memory
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(x_d));

    std::string implementation_name = uut.version_name;

    // compare the result
    bool test_pass = vector_compare(solution, x);
    if (not test_pass){
        std::cout << "SymGS mini test failed" << std::endl;
    }
    return test_pass;
}

bool test_SymGS(
    HPCG_functions<double> &baseline, HPCG_functions<double> &uut,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d
    )

{   

    int num_rows = A.get_num_rows();
    // since symGS changes x, we preserve the original x
    std::vector<double> x(num_rows, 0.0);
    std::vector<double> uut_result(num_rows, 0.0);
    std::vector<double> baseline_result(num_rows, 0.0);
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    uut.compute_SymGS(A, x_d, y_d);

    // get the result back
    CHECK_CUDA(cudaMemcpy(uut_result.data(), x_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    // testing is either done by a comparison of the results or by checking the relative residual
    // this depends on the version

    bool test_pass = true;

    if(uut.norm_based){
        double rr_norm = relative_residual_norm_for_SymGS(
            A,
            x_d, y_d
        );

        int nx = A.get_nx();
        int ny = A.get_ny();
        int nz = A.get_nz();

        double threshold_norm = uut.getSymGS_rrNorm_zero_init(nx, ny, nz);
        
        test_pass = rr_norm <= threshold_norm;

        if (not test_pass){
            std::cout << "SymGS test failed for size " << nx << "x" << ny << "x" << nz << std::endl;
            std::cout << "The rr norm was " << rr_norm << " and the threshold was " << threshold_norm << std::endl;
            // test_pass = false;
        }
        // else {
        //     std::cout << "SymGS test passed for size " << nx << "x" << ny << "x" << nz << std::endl;
        //     std::cout << "The rr norm was " << rr_norm << " and the threshold was " << threshold_norm << std::endl;
        // }
    } else{
        // run the baseline
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
        baseline.compute_SymGS(A, x_d, y_d);

        // get the result back
        CHECK_CUDA(cudaMemcpy(baseline_result.data(), x_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

        test_pass = vector_compare(uut_result, baseline_result);
        if (not test_pass){
            std::cout << "SymGS test failed" << std::endl;
        }
    }

    // copy the original x back
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    return test_pass;
}

bool test_SymGS(
    HPCG_functions<double>& baseline, HPCG_functions<double>& uut,
    striped_Matrix<double> & striped_A, // we pass A for the metadata
    
    double * y_d // the vectors x is already on the device
        
){
    // std::cout << "SymGS test 2" << std::endl;

    int num_rows = striped_A.get_num_rows();

    sparse_CSR_Matrix<double>* A = striped_A.get_CSR();
    // A.sparse_CSR_Matrix_from_striped(striped_A);

    int num_rows_baseline = A->get_num_rows();

    double * x_uut_d;
    CHECK_CUDA(cudaMalloc(&x_uut_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMemset(x_uut_d, 0, num_rows * sizeof(double)));

    // std::cout << "Baseline name = " << baseline.version_name << std::endl;

    std::vector<double> x_baseline(num_rows, 0.0);
    std::vector<double> x_uut(num_rows, 0.0);

    // in case it is a norm based SymGS (we do more than one iteration)
    // we need to store and adjust the number of max iterations
    int original_max_iter = uut.get_maxSymGSIters();
    uut.set_maxSymGSIters(500);

    // std::vector<double> looki(5);
    // CHECK_CUDA(cudaMemcpy(looki.data(), x_uut_d, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "Looki: " << looki[0] << std::endl;

    uut.compute_SymGS(striped_A,
                    x_uut_d, y_d);


    // testing depends on the version it either checks for a relative residual or for the result

    bool test_pass;

    if (uut.norm_based){

        // std::cout << "SymGS norm based test" << std::endl;
        // std::cout << "version name " << uut.version_name << std::endl;

        double rr_norm = relative_residual_norm_for_SymGS(
            striped_A,
            x_uut_d, y_d
        );

        int nx = A->get_nx();
        int ny = A->get_ny();
        int nz = A->get_nz();

        double threshold_norm = uut.getSymGS_rrNorm_zero_init(nx, ny, nz);
        
        test_pass = rr_norm <= threshold_norm;

        if (not test_pass){
            std::cout << "SymGS test failed for size " << nx << "x" << ny << "x" << nz << " version: " << uut.version_name << std::endl;
            std::cout << "The rr norm was " << rr_norm << " and the threshold is " << threshold_norm << std::endl;
            // std::cout << "heul doch" << std::endl;
            // test_pass = false;
        }
        // else {
        //     std::cout << "SymGS test passed for size " << nx << "x" << ny << "x" << nz << std::endl;
        //     std::cout << "The rr norm was " << rr_norm << " and the threshold was " << threshold_norm << std::endl;
        // }

    } else{
        // this is the case where we compare to a baseline
        double * x_baseline_d;

        CHECK_CUDA(cudaMalloc(&x_baseline_d, num_rows * sizeof(double)));

        // we need the x to be all set to zero, otherwise with different initial conditions the results will be different
        CHECK_CUDA(cudaMemset(x_baseline_d, 0, num_rows * sizeof(double)));

        baseline.compute_SymGS(*A, x_baseline_d, y_d);

        // and now we need to copy the result back and de-allocate the memory
        CHECK_CUDA(cudaMemcpy(x_baseline.data(), x_baseline_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(x_uut.data(), x_uut_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
        
        CHECK_CUDA(cudaFree(x_baseline_d));

        // compare the results
        test_pass = vector_compare(x_baseline, x_uut);
        
        if (not test_pass){
            std::cout << "SymGS test failed for uut: " << uut.version_name << std::endl;
            // std::cout << "I am sad" << std::endl;
        }
        // std::cout << "Baseline: " << x_baseline[0] << std::endl;
        // std::cout << "UUT: " << x_uut[0] << std::endl;
    }

    CHECK_CUDA(cudaFree(x_uut_d));

    // reset the number of iterations
    uut.set_maxSymGSIters(original_max_iter);
    
    return test_pass;
}
