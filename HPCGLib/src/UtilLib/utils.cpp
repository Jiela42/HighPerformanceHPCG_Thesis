#include "UtilLib/utils.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"

#include <cmath>

bool double_compare(double a, double b){
    return abs(a - b) < error_tolerance;
}

bool vector_compare(const std::vector<double>& a, const std::vector<double>& b){
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match" << std::endl;
        return false;
    }
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

bool vector_compare(const std::vector<int>& a, const std::vector<int>& b, std::string info){
    bool test_result = true;
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match for " << info << " size a = " << a.size() << " size b = " << b.size() << std::endl;
         test_result = false;
    }

    int fault_ctr = 0;

    for(int i = 0; i < a.size(); i++){
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

    std::vector<int> row_ptr = A.get_row_ptr();
    std::vector<int> col_idx = A.get_col_idx();
    std::vector<double> values = A.get_values();
    std::vector<double> Ax(x_solution.size(), 0.0);

    // first calculate Ax
    for (int i = 0; i < row_ptr.size() - 1; i++){
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            Ax[i] += values[j] * x_solution[col_idx[j]];
        }
    }

    // now calculate the difference and the L2 norm
    double L2_norm = 0.0;

    for (int i = 0; i < x_solution.size(); i++){
        L2_norm += pow(Ax[i] - true_solution[i], 2);
    }

    return sqrt(L2_norm);
}

double relative_residual_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution){
    std::vector<int> row_ptr = A.get_row_ptr();
    std::vector<int> col_idx = A.get_col_idx();
    std::vector<double> values = A.get_values();
    std::vector<double> Ax(x_solution.size(), 0.0);

    // First calculate Ax
    for (int i = 0; i < row_ptr.size() - 1; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            Ax[i] += values[j] * x_solution[col_idx[j]];
        }
    }

    // Now calculate the difference and the L2 norm
    double L2_norm = 0.0;
    for (int i = 0; i < x_solution.size(); i++) {
        L2_norm += pow(Ax[i] - true_solution[i], 2);
    }
    L2_norm = sqrt(L2_norm);

    // Calculate the L2 norm of the true solution
    double L2_norm_true = 0.0;
    for (int i = 0; i < true_solution.size(); i++) {
        L2_norm_true += pow(true_solution[i], 2);
    }
    L2_norm_true = sqrt(L2_norm_true);

    // Return the relative residual norm
    return L2_norm / L2_norm_true;
}