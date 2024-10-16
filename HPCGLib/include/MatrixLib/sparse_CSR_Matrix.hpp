#ifndef SPARSE_CSR_MATRIX_HPP
#define SPARSE_CSR_MATRIX_HPP

#include <vector>

template <typename T>
class sparse_CSR_Matrix {
public:
    sparse_CSR_Matrix();
    sparse_CSR_Matrix(int nx, int ny, int nz, int nnz, T* vals, int* row_ptr, int* col_idx);

    const std::vector<int>& get_row_ptr() const;
    const std::vector<int>& get_col_idx() const;
    const std::vector<T>& get_values() const;
    int get_num_rows() const;
    int get_num_cols() const;
    int get_nx() const;
    int get_ny() const;
    int get_nz() const;
    int get_nnz() const;
    T get_element(int i, int j) const;


private:
    int nx;
    int ny;
    int nz;
    int nnz;
    int num_rows;
    int num_cols;
    std::vector<T> row_ptr;
    std::vector<T> col_idx;
    std::vector<T> values;
};

endif // SPARSE_CSR_MATRIX_HPP
