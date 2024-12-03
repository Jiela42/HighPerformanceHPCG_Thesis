#ifndef BANDED_MATRIX_HPP
#define BANDED_MATRIX_HPP

#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/matrix_basics.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cassert>

// Forward declaration of sparse_CSR_Matrix
template <typename T>
class sparse_CSR_Matrix;

template <typename T>
class banded_Matrix {
    public:
        banded_Matrix();
        void banded_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T> A);

        int get_num_rows() const;
        int get_num_cols() const;
        int get_num_bands() const;
        int get_nx() const;
        int get_ny() const;
        int get_nz() const;
        int get_nnz() const;
        MatrixType get_matrix_type() const;
        T get_element(int i, int j) const;
        std::vector <int>& get_j_min_i();
        std::vector <T>& get_values();
        void set_num_rows(int num_rows);
        void print() const;
        // void compare_to(banded_Matrix<T>& other) const;
        // void write_to_file() const;


    private:
        int nx;
        int ny;
        int nz;
        int nnz;
        int num_rows;
        int num_cols;
        int num_bands;
        std::vector<int> j_min_i;
        std::vector<T> values;
        MatrixType matrix_type;
        void banded_3D27P_Matrix_from_CSR(sparse_CSR_Matrix<T> A);

};

#endif // BANDED_MATRIX_HPP