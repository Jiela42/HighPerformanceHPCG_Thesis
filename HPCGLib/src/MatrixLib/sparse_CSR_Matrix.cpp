#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>


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

    this->row_ptr = std::vector<int>(row_ptr, row_ptr + this->num_rows + 1);
    this->col_idx = std::vector<int>(col_idx, col_idx + this->nnz);
    this->values = std::vector<T>(vals, vals + this->nnz);

}

template <typename T>
sparse_CSR_Matrix<T>::sparse_CSR_Matrix(int nx, int ny, int nz, int nnz, std::vector<T> vals, std::vector<int> row_ptr, std::vector<int> col_idx) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nnz = nnz;
    this->num_rows = nx * ny * nz;
    this->num_cols = nx * ny * nz;

    this->row_ptr = row_ptr;
    this->col_idx = col_idx;
    this->values = vals;
}

template <typename T>
const std::vector<int>& sparse_CSR_Matrix<T>::get_row_ptr(){
    return this->row_ptr;
}

template <typename T>
const std::vector<int>& sparse_CSR_Matrix<T>::get_col_idx(){
    return this->col_idx;
}

template <typename T>
const std::vector<T>& sparse_CSR_Matrix<T>::get_values(){
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
    printf("WARNING Element row %d, col %d not found in sparse CSR Matrix\n", row, col);
    return T();
}

template <typename T>
void sparse_CSR_Matrix<T>::print() const{
    std::cout << "Row Pointer: ";
    for (int i = 0; i < this->num_rows + 1; i++) {
        std::cout << this->row_ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Column Index: ";
    for (int i = 0; i < this->nnz; i++) {
        std::cout << this->col_idx[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Values: ";
    for (int i = 0; i < this->nnz; i++) {
        std::cout << this->values[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void sparse_CSR_Matrix<T>::compare_to(sparse_CSR_Matrix<T>& other) const{

    bool same = true;

    if (this->num_rows != other.get_num_rows()){
        printf("Matrices have different number of rows: this has %d the other %d\n", this->num_rows, other.get_num_rows());
        same = false;
    }
    if (this->num_cols != other.get_num_cols()){
        printf("Matrices have different number of cols: this has %d the other %d\n", this->num_cols, other.get_num_cols());
        same = false;
    }

    for (int i = 0; i < this->num_rows; i++) {
        int start = this->row_ptr[i];
        int end = this->row_ptr[i + 1];
        int other_start = other.get_row_ptr()[i];
        int other_end = other.get_row_ptr()[i + 1];
        if (end - start != other_end - other_start) {
            printf("Row %d has different number of non-zero elements\n", i);
            same = false;
        }
        for (int j = start; j < end; j++) {
            if (this->col_idx[j] != other.get_col_idx()[j] || this->values[j] != other.get_values()[j]) {
                printf("Element at row %d, col %d is different\n", i, this->col_idx[j]);
                same = false;
            }
        }
    }
    if (same){
        printf("Matrices are the same\n");
    }
}

//explicit instantiation
template class sparse_CSR_Matrix<double>;

