#ifndef CUSPARSE_HPP
#define CUSPARSE_HPP

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <iostream>

template <typename T>
class cuSparse_Implementation : public HPCG_functions<T> {
public:
    void compute_SPMV(const sparse_CSR_Matrix<T>& A, const std::vector<T>& x, std::vector<T>& y) override {
        cusparse_computeSPMV(A, x, y);
    }

    void compute_CG(const sparse_CSR_Matrix<T>& A, const std::vector<T>& b, std::vector<T>& x) override {
        std::cerr << "Warning: compute_CG is not implemented in cuSparse_Implementation." << std::endl;
    }

    void compute_MG(const sparse_CSR_Matrix<T>& A, const std::vector<T>& x, std::vector<T>& y) override {
        std::cerr << "Warning: compute_MG is not implemented in cuSparse_Implementation." << std::endl;
    }

    void compute_WAXPBY(const std::vector<T>& x, const std::vector<T>& y, std::vector<T>& w, T alpha, T beta) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in cuSparse_Implementation." << std::endl;
    }

    void compute_Dot(const std::vector<T>& x, const std::vector<T>& y, T& result) override {
        std::cerr << "Warning: compute_Dot is not implemented in cuSparse_Implementation." << std::endl;
    }
private:
    void cusparse_computeSPMV(const sparse_CSR_Matrix<T>& A, const std::vector<T>& x, std::vector<T>& y);
};

#endif // CUSPARSE_HPP