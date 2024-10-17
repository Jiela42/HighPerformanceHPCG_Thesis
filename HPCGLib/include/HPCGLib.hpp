#ifndef HPCGLIB_HPP
#define HPCGLIB_HPP

#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>

// define number of iterations we want to have
#define num_bench_iter 10

template <typename T>
class HPCG_functions {
    public:
        virtual void compute_CG(const sparse_CSR_Matrix<T> & A, const std::vector<T> & b, std::vector<T> & x) = 0;
        virtual void compute_MG(const sparse_CSR_Matrix<T> & A, const std::vector<T> & x, std::vector<T> & y) = 0;
        virtual void compute_SPMV(const sparse_CSR_Matrix<T> & A, const std::vector<T> & x, std::vector<T> & y) = 0;
        virtual void compute_WAXPBY(const std::vector<T> & x, const std::vector<T> & y, std::vector<T> & w, const T alpha, const T beta) = 0;
        virtual void compute_Dot(const std::vector<T> & x, const std::vector<T> & y, T & result) = 0;
private:
    int getNumber() const {
        return num_bench_iter;
    }
};

#endif // HPCGLIB_HPP