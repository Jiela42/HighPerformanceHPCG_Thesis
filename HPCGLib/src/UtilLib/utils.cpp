#include "UtilLib/utils.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"

#include <cmath>

bool double_compare(double a, double b){

    if(std::isnan(a) or std::isnan(b)){
        std::cout << "One of the numbers is nan" << std::endl;
        return false;
    }
    if(std::abs(a - b) > error_tolerance){
        std::cout << "Error: " << a << " != " << b << std::endl;
        std::cout << "Difference: " << std::abs(a - b) << std::endl;
    }

    return std::abs(a - b) < error_tolerance;
}

bool relaxed_double_compare(double a, double b, double tolerance){
    // this function is for handling small issues like the kind that arise from exploiting commutativity on floats
    if(std::isnan(a) or std::isnan(b)){
        std::cout << "One of the numbers is nan" << std::endl;
        return false;
    }
    if(std::abs(a - b) >= tolerance){
        std::cout << "Error: " << a << " != " << b << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Difference: " << std::abs(a - b) << std::endl;
    }
    
    return std::abs(a - b) < tolerance;
}

bool vector_compare(const std::vector<double>& a, const std::vector<double>& b){
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match" << std::endl;
        return false;
    }
    int error_ctr = 0;
    for (int i = 0; i < a.size(); i++){
        if(std::isnan(a[i]) or std::isnan(b[i])){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << std::endl;
            return false;
        }
        if (not double_compare (a[i], b[i])){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename t>
bool vector_compare(const std::vector<t>& a, const std::vector<t>& b, std::string info){
    bool test_result = true;
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match for " << info << " size a = " << a.size() << " size b = " << b.size() << std::endl;
         test_result = false;
    }

    int fault_ctr = 0;

    for(local_int_t i = 0; i < a.size(); i++){
        if (a[i] != b[i] && fault_ctr < 10){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << " for " << info << std::endl;
            test_result = false;
            fault_ctr++;
        }
    }
    return test_result;
}

bool vector_compare(const std::vector<local_int_t>& a, const std::vector<local_int_t>& b, std::string info){
    bool test_result = true;
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match for " << info << " size a = " << a.size() << " size b = " << b.size() << std::endl;
         test_result = false;
    }

    int fault_ctr = 0;

    for(local_int_t i = 0; i < a.size(); i++){
        if (a[i] != b[i] && fault_ctr < 10){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << " for " << info << std::endl;
            test_result = false;
            fault_ctr++;
        }
    }
    return test_result;
}


double L2_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution){

    // this thing expects A, to be on the gpu

    std::vector<local_int_t> row_ptr = A.get_row_ptr();
    std::vector<local_int_t> col_idx = A.get_col_idx();
    std::vector<double> values = A.get_values();
    std::vector<double> Ax(x_solution.size(), 0.0);

    // first calculate Ax
    for (local_int_t i = 0; i < row_ptr.size() - 1; i++){
        for (local_int_t j = row_ptr[i]; j < row_ptr[i+1]; j++){
            Ax[i] += values[j] * x_solution[col_idx[j]];
        }
    }

    // now calculate the difference and the L2 norm
    double L2_norm = 0.0;

    for (local_int_t i = 0; i < x_solution.size(); i++){
        L2_norm += pow(Ax[i] - true_solution[i], 2);
    }

    return sqrt(L2_norm);
}

double relative_residual_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution){
    std::vector<local_int_t> row_ptr = A.get_row_ptr();
    std::vector<local_int_t> col_idx = A.get_col_idx();
    std::vector<double> values = A.get_values();
    std::vector<double> Ax(x_solution.size(), 0.0);

    // First calculate Ax
    for (local_int_t i = 0; i < row_ptr.size() - 1; i++) {
        for (local_int_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            Ax[i] += values[j] * x_solution[col_idx[j]];
        }
    }

    // Now calculate the difference and the L2 norm
    double L2_norm = 0.0;
    for (local_int_t i = 0; i < x_solution.size(); i++) {
        L2_norm += pow(Ax[i] - true_solution[i], 2);
    }
    L2_norm = sqrt(L2_norm);

    // Calculate the L2 norm of the true solution
    double L2_norm_true = 0.0;
    for (local_int_t i = 0; i < true_solution.size(); i++) {
        L2_norm_true += pow(true_solution[i], 2);
    }
    L2_norm_true = sqrt(L2_norm_true);

    // Return the relative residual norm
    return L2_norm / L2_norm_true;
}

void sanity_check_vector(std::vector<double>& a, std::vector<double>& b){
    assert(a.size() == b.size());
    for (local_int_t i = 0; i < a.size(); i++){
        assert(double_compare(a[i], b[i]));
    }
}

void sanity_check_vectors(std::vector<double *>& device, std::vector<std::vector<double>>& original){
    assert(device.size() == original.size());

    for(local_int_t i = 0; i < device.size(); i++){
        // std::cout << "checking vector " << i << std::endl;
        std::vector<double> host(original[i].size());
        CHECK_CUDA(cudaMemcpy(host.data(), device[i], original[i].size() * sizeof(double), cudaMemcpyDeviceToHost));
        sanity_check_vector(host, original[i]);
    }
}

template bool vector_compare<int>(const std::vector<int>&, const std::vector<int>&, std::string);
template bool vector_compare<long>(const std::vector<long>&, const std::vector<long>&, std::string);
template bool vector_compare<long long>(const std::vector<long long>&, const std::vector<long long>&, std::string);
