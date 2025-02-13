#include "testing.hpp"

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveStriped.cuh"

#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.hpp"

#include <iostream>

// these functions are really just wrappers to check for correctness
// hence the only thing the function needs to allcoate is space for the outputs
// depending on the versions they may require different inputs, hence the method overloading

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
    
    sparse_CSR_Matrix<double> A;
    A.sparse_CSR_Matrix_from_striped(striped_A);

    int num_rows_baseline = A.get_num_rows();
    std::vector<double> y_baseline(num_rows, 0.0);
    std::vector<double> y_uut(num_rows, 0.0);

    double * y_baseline_d;
    double * y_uut_d;

    CHECK_CUDA(cudaMalloc(&y_baseline_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_uut_d, num_rows * sizeof(double)));

    baseline.compute_SPMV(A,
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
        std::cout << "Dot product failed: baseline = " << result_baseline << " uut = " << result_uut << std::endl;
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

    // srand(RANDOM_SEED);

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

    bool test_pass = double_compare(result, result_uut);
    if (not test_pass){
        std::cout << "Dot product failed: baseline = " << result << " uut = " << result_uut << std::endl;
    }
    // else {
    //     std::cout << "Dot product passed: baseline = " << result << " uut = " << result_uut << std::endl;
    // }

    return test_pass;
}

bool test_SymGS(
    HPCG_functions<double>&uut,
    sparse_CSR_Matrix<double> & A
    ){
    // This is the mini test for the SymGS function

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

    int num_rows = striped_A.get_num_rows();

    sparse_CSR_Matrix<double> A;
    A.sparse_CSR_Matrix_from_striped(striped_A);

    int num_rows_baseline = A.get_num_rows();


    double * x_uut_d;
    CHECK_CUDA(cudaMalloc(&x_uut_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMemset(x_uut_d, 0, num_rows * sizeof(double)));


    std::vector<double> x_baseline(num_rows, 0.0);
    std::vector<double> x_uut(num_rows, 0.0);

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

        int nx = A.get_nx();
        int ny = A.get_ny();
        int nz = A.get_nz();

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

        baseline.compute_SymGS(A,
                            x_baseline_d, y_d);

        
        // and now we need to copy the result back and de-allocate the memory
        CHECK_CUDA(cudaMemcpy(x_baseline.data(), x_baseline_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(x_uut.data(), x_uut_d, num_rows * sizeof(double), cudaMemcpyDeviceToHost));
        
        CHECK_CUDA(cudaFree(x_baseline_d));

        // compare the results
        test_pass = vector_compare(x_baseline, x_uut);
        
        if (not test_pass){
            std::cout << "SymGS test failed for uut: " << uut.version_name << std::endl;
        }
        // std::cout << "Baseline: " << x_baseline[0] << std::endl;
        // std::cout << "UUT: " << x_uut[0] << std::endl;
    }

    CHECK_CUDA(cudaFree(x_uut_d));

    
    return test_pass;
}
