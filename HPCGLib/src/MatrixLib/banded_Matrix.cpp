#include "MatrixLib/banded_Matrix.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"

#include <vector>
#include <iostream>
#include <string>

// #include <stdio.h>

template <typename T>
banded_Matrix<T>::banded_Matrix() {
    this->nx = 0;
    this->ny = 0;
    this->nz = 0;
    this->nnz = 0;
    this->matrix_type = MatrixType::UNKNOWN;
    this->num_rows = 0;
    this->num_cols = 0;
    this->num_bands = 0;
    this->j_min_i.clear();
    this->values.clear();
}

template <typename T>
void banded_Matrix<T>::banded_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T> A){
    if (A.get_matrix_type() == MatrixType::Stencil_3D27P) {
        this->banded_3D27P_Matrix_from_CSR(A);
    } else {
        printf("ERROR: Unsupported matrix type for conversion to banded matrix\n");
        exit(1);
    }
}

template <typename T>
void banded_Matrix<T>::banded_3D27P_Matrix_from_CSR(sparse_CSR_Matrix<T> A){
    
    assert(A.get_matrix_type() == MatrixType::Stencil_3D27P);
    this->matrix_type = MatrixType::Stencil_3D27P;
    this->nx = A.get_nx();
    this->ny = A.get_ny();
    this->nz = A.get_nz();
    this->nnz = A.get_nnz();
    this->num_rows = A.get_num_rows();
    this->num_cols = A.get_num_cols();
    this->num_bands = 27;
    this->j_min_i = std::vector<int>(this->num_bands, 0);
    this->values = std::vector<T>(this->num_bands * this->num_rows, 0);

    // first we make our mapping for the j_min_i
    // each point has num_band neighbours and each is associated with a coordinate relative to the point
    // the point itself is a neighobour, too {0,0,0}
    int neighbour_offsets [num_bands][3] = {
        {-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
        {-1, 0, -1}, {0, 0, -1}, {1, 0, -1},
        {-1, 1, -1}, {0, 1, -1}, {1, 1, -1},
        {-1, -1, 0}, {0, -1, 0}, {1, -1, 0},
        {-1, 0, 0}, {0, 0, 0}, {1, 0, 0},
        {-1, 1, 0}, {0, 1, 0}, {1, 1, 0},
        {-1, -1, 1}, {0, -1, 1}, {1, -1, 1},
        {-1, 0, 1}, {0, 0, 1}, {1, 0, 1},
        {-1, 1, 1}, {0, 1, 1}, {1, 1, 1}
    };

    for (int i = 0; i < this->num_bands; i++) {

        int off_x = neighbour_offsets[i][0];
        int off_y = neighbour_offsets[i][1];
        int off_z = neighbour_offsets[i][2];
        
        this->j_min_i[i] = off_x + off_y * this->nx + off_z * this->nx * this->ny;
    }

    int elem_ctr = 0;

    // now that we have the static offsets which define i & j, we can make the actual matrix
    for (int i = 0; i < this->num_rows; i++) {
        for (int band_j = 0; band_j < this->num_bands; band_j++) {
            int j = this->j_min_i[band_j] + i;
            // check if j is in bounds (since not every point has all 27 neighbours)
            if (j >= 0 && j < this->num_cols) {
                double elem = A.get_element(i, j);
                // also make sure we don't add zero elements
                if(elem != 0){
                    this->values[i * this->num_bands + band_j] = elem;
                    elem_ctr++;
                }
            }
        }
    }
    assert(elem_ctr == this->nnz);
}

template <typename T>
int banded_Matrix<T>::get_num_rows() const{
    return this->num_rows;
}

template <typename T>
int banded_Matrix<T>::get_num_cols() const{
    return this->num_cols;
}

template <typename T>
int banded_Matrix<T>::get_num_bands() const{
    return this->num_bands;
}

template <typename T>
int banded_Matrix<T>::get_nx() const{
    return this->nx;
}

template <typename T>
int banded_Matrix<T>::get_ny() const{
    return this->ny;
}

template <typename T>
int banded_Matrix<T>::get_nz() const{
    return this->nz;
}


template <typename T>
int banded_Matrix<T>::get_nnz() const{
    return this->nnz;
}

template <typename T>
MatrixType banded_Matrix<T>::get_matrix_type() const{
    return this->matrix_type;
}

template <typename T>
std::vector<int>& banded_Matrix<T>::get_j_min_i(){
    return this->j_min_i;
}

template <typename T>
std::vector<T>& banded_Matrix<T>::get_values(){
    return this->values;
}

template <typename T>
T banded_Matrix<T>::get_element(int i, int j) const{

    // check if j is in bounds (since not every point has all 27 neighbours)
    if (j >= 0 && j < this->num_cols) {
        for (int band_j = 0; band_j < this->num_bands; band_j++) {
            if (j == i + this->j_min_i[band_j]) {
                return this->values[i * this->num_bands + band_j];
            }
        }
    }
    // printf("WARNING Element row %d, col %d not found\n", i, j);
    return T();
}

template <typename T>
void banded_Matrix<T>::set_num_rows(int num_rows){
    this->num_rows = num_rows;
}

template <typename T>
void banded_Matrix<T>::print() const{
    std::cout << "Banded Matrix: " << std::endl;
    std::cout << "nx: " << this->nx << " ny: " << this->ny << " nz: " << this->nz << std::endl;
    std::cout << "num_rows: " << this->num_rows << " num_cols: " << this->num_cols << std::endl;
    std::cout << "num_bands: " << this->num_bands << std::endl;
    std::cout << "j_min_i: ";
    for (int i = 0; i < this->num_bands; i++) {
        std::cout << this->j_min_i[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Values: ";
    for (int i = 0; i < this->num_rows; i++) {
        for (int j = 0; j < this->num_bands; j++) {
            std::cout << this->values[i * this->num_bands + j] << " ";
        }
        std::cout << std::endl;
    }
}

// explicit template instantiation
template class banded_Matrix<double>;