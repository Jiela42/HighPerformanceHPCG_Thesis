#ifndef HPCGLIB_HPP
#define HPCGLIB_HPP

#include sparse_CSR_Matrix.hpp
#include <vector>

template <typename T>
class HPCGLib {
    public:
        virtual void CG(const sparse_CSR_Matrix & A, const std::vector & b, std::vector & x) = 0;
        virtual void MG(const sparse_CSR_Matrix & A, const std::vector & x, std::vector & y) = 0;
        virtual void SPMV(const sparse_CSR_Matrix & A, const std::vector & x, std::vector & y) = 0;
        virtual void WAXPBY(const std::vector & x, const std::vector & y, std::vector & w, const T alpha, const T beta) = 0;
        virtual void Dot(const std::vector & x, const std::vector & y, T & result) = 0;
};

#endif // HPCGLIB_HPP