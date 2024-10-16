#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <iostream>
#include <stdio.h>


template <typename T>
sparse_CSR_Matrix<T>::sparse_CSR_Matrix() {
    this->nx = 0;
    this->ny = 0;
    this->nz = 0;
    this->nnz = 0;
    this->num_rows = 0;
    this->num_cols = 0;
    this->row_ptr.clear();
    this->col_idx.clear();
    this->values.clear();
}

template <typename T>
sparse_CSR_Matrix<T>::sparse_CSR_Matrix(int nx, int ny, int nz, int nnz, T* vals, int* row_ptr, int* col_idx) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nnz = nnz;
    this->num_rows = nx * ny * nz;
    this->num_cols = nx * ny * nz;

    this->row_ptr = std::vector<T>(row_ptr, row_ptr + this->num_rows + 1);
    this->col_idx = std::vector<T>(col_idx, col_idx + this->nnz);
    this->values = std::vector<T>(vals, vals + this->nnz);

}

template <typename T>
const std::vector<int>& sparse_CSR_Matrix<T>::get_row_ptr() const{
    return this->row_ptr;
}

template <typename T>
const std::vector<int>& sparse_CSR_Matrix<T>::get_col_idx() const{
    return this->col_idx;
}

template <typename T>
const std::vector<T>& sparse_CSR_Matrix<T>::get_values() const{
    return this->values;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_num_rows() const{
    return this->num_rows;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_num_cols() const{
    return this->num_cols;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_nx() const{
    return this->nx;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_ny() const{
    return this->ny;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_nz() const{
    return this->nz;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_nnz() const{
    return this->nnz;
}

template <typename T>
T sparse_CSR_Matrix<T>::get_element(int row, int col) const{
    
    int start = this->row_ptr[row];
    int end = this->row_ptr[row + 1];
    for (int i = start; i < end; i++) {
        if (this->col_idx[i] == col) {
            return this->values[i];
        }
    }
    printf("WARNING Element row %d, col %d not found\n", row, col);
    return T();
}

//explicit instantiation
template class sparse_CSR_Matrix<double>;

