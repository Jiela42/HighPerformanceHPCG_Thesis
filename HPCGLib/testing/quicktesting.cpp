// just hello world to check if cmake setup works
#include <iostream>
#include "MatrixLib/generations.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "HPCG_versions/cusparse.hpp"

int main() {

    int num = 4;

    std::cout << "Hello, World!" << std::endl;
    std::cout << "Starting Matrix Generation" << std::endl;
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(num, num, num);
    
    // call spmv
    std::vector<double> x(num*num*num, 1.0);
    std::vector<double> y(num*num*num, 0.0);

    cuSparse_Implementation<double> cuSparse;
    cuSparse.compute_SPMV(problem.first, x, y);
    

    std::cout << "Matrix Generation Complete" << std::endl;
    return 0;
}