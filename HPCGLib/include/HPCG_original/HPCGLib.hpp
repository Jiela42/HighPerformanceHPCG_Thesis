#ifndef HPCGLIB_HPP
#define HPCGLIB_HPP

template <typename T>
class HPCG {
    public:
        virtual void CG(const SparseMatrix & A, const Vector & b, Vector & x) = 0;
        virtual void MG(const SparseMatrix & A, const Vector & x, Vector & y) = 0;
        virtual void SPMV(const SparseMatrix & A, const Vector & x, Vector & y) = 0;
        virtual void WAXPBY(const Vector & x, const Vector & y, Vector & w, const T alpha, const T beta) = 0;
        virtual void Dot(const Vector & x, const Vector & y, T & result) = 0;

#endif