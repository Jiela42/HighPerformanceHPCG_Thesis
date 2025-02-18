#ifndef STRIPED_MATRIX_HPP
#define STRIPED_MATRIX_HPP

#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/matrix_basics.hpp"
#include "MatrixLib/generations.cuh"

#include <vector>
#include <iostream>
#include <string>
#include <cassert>

// Forward declaration of sparse_CSR_Matrix
template <typename T>
class sparse_CSR_Matrix;

template <typename T>
class striped_Matrix {
    public:
        striped_Matrix();
        ~striped_Matrix();
        void striped_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T> & A);

        void generate_coloring();

        void copy_Matrix_toGPU();
        void copy_Matrix_toCPU();

        void remove_Matrix_from_GPU();

        int *get_color_pointer_d();
        int *get_color_sorted_rows_d();
        std::vector<int> get_color_pointer_vector();
        std::vector<int> get_color_sorted_rows_vector();
        
        int get_num_rows() const;
        int get_num_cols() const;
        int get_num_stripes() const;
        int get_nx() const;
        int get_ny() const;
        int get_nz() const;
        int get_nnz() const;
        int get_diag_index() const;
        MatrixType get_matrix_type() const;
        T get_element(int i, int j) const;
        std::vector <int>& get_j_min_i();
        std::vector <T>& get_values();
        int * get_j_min_i_d();
        T * get_values_d();
        void set_num_rows(int num_rows);
        void print_COR_Format();
        void print() const;
        // void compare_to(striped_Matrix<T>& other) const;
        // void write_to_file() const;


    private:
        int nx;
        int ny;
        int nz;
        int nnz;
        int diag_index;
        int num_rows;
        int num_cols;
        int num_stripes;
        std::vector<int> j_min_i;
        std::vector<T> values;
        MatrixType matrix_type;
        void striped_3D27P_Matrix_from_CSR(sparse_CSR_Matrix<T> & A);
        void striped_3D27P_Matrix_from_CSR_onGPU(sparse_CSR_Matrix<T> & A);
        int *j_min_i_d;
        T *values_d;
        int* color_pointer_d;
        int* color_sorted_rows_d;
};

#endif // STRIPED_MATRIX_HPP