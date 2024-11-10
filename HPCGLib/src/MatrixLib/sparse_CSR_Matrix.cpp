#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>


template <typename T>
sparse_CSR_Matrix<T>::sparse_CSR_Matrix() {
    
    this->matrix_type = MatrixType::UNKNOWN;

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
sparse_CSR_Matrix<T>::sparse_CSR_Matrix(int nx, int ny, int nz, int nnz, MatrixType mt, T* vals, int* row_ptr, int* col_idx) {
    
    this->matrix_type = mt;
    
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
sparse_CSR_Matrix<T>::sparse_CSR_Matrix(int nx, int ny, int nz, int nnz, MatrixType mt, std::vector<T> vals, std::vector<int> row_ptr, std::vector<int> col_idx) {
    
    this->matrix_type = mt;

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
void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_banded(banded_Matrix<T> A){

    if(A.get_matrix_type() == MatrixType::Stencil_3D27P){
        assert(A.get_num_bands() == 27);
        assert(A.get_num_rows() == A.get_num_cols());
        assert(A.get_num_rows() == A.get_nx() * A.get_ny() * A.get_nz());
    }
    this->sparse_CSR_Matrix_from_banded_transformation(A);
}

template <typename T>
void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_banded_transformation(banded_Matrix<T> A){

    this->matrix_type = A.get_matrix_type();
    this->nx = A.get_nx();
    this->ny = A.get_ny();
    this->nz = A.get_nz();
    this->nnz = A.get_nnz();
    this->num_rows = A.get_num_rows();
    this->num_cols = A.get_num_cols();

    assert(A.get_num_bands() == A.get_j_min_i().size());

    this->row_ptr = std::vector<int>(this->num_rows + 1, 0);
    this->col_idx = std::vector<int>(this->nnz, 0);
    this->values = std::vector<T>(this->nnz, 0);

    // for(int i =0; i< A.get_num_bands(); i++){
    //     std::cout << A.get_values()[i] << std::endl;
    // }

    std::vector<T> banded_vals = A.get_values();

    // for(int i=0; i< A.get_num_bands(); i++){
    //     std::cout << "banded_vals: " << banded_vals[i] << std::endl;
    // }
    int elem_count = 0;

    for(int i = 0; i < this->num_rows; i++){
        int nnz_i = 0;
        for(int band_j = 0; band_j < A.get_num_bands(); band_j++){
            int j = A.get_j_min_i()[band_j] + i;
            double val = banded_vals[i*A.get_num_bands() + band_j];
            // int val = A.get_element(i, j);
            // if(elem_count == 0 && i == 0){
            //     std::cout << "val: " << val << std::endl;
            // }
            if((val!= 0.0)){
                this->col_idx[elem_count] = j;
                this->values[elem_count] = val;
                elem_count++;
            }
            else{
                // if (i == 0){
                // std::cout << "val is zero: val: " << val << std::endl;
                // }
            }

            this->row_ptr[i + 1] = elem_count;
        }
    }
    // std::cout << "elem_count: " << elem_count << std::endl;
    // std::cout << "nnz: " << this->nnz << std::endl;
    assert(elem_count == this->nnz);

}

template <typename T>
void sparse_CSR_Matrix<T>::sanity_check_3D27P(){
    assert(this->matrix_type == MatrixType::Stencil_3D27P);
    assert(this->num_rows == this->num_cols);
    assert(this->num_rows == this->nx * this->ny * this->nz);
    assert(this->row_ptr.size() == this->num_rows + 1);
    assert(this->col_idx.size() == this->nnz);
    assert(this->nnz == this->row_ptr[this->num_rows]);
}

template <typename T>
void sparse_CSR_Matrix<T>::iterative_values(){

    double val = 0.1;

    for (int i = 0; i < this->nnz; i++) {
        this->values[i] = val;
        val += 0.1;
        if (val > 10.0) {
            val = 0.1;
        }
        // if(val < 0.1){
        //     std::cout << "val is zero" << std::endl;
        // }
    }
}

template <typename T>
void sparse_CSR_Matrix<T>::random_values(int seed){
    srand(seed);
    for (int i = 0; i < this->nnz; i++) {
        T val = (T)rand() / RAND_MAX;
        // if(i==0){
        //     std::cout << "val: " << val << std::endl;
        // }
        this->values[i] = val;
    }
    // std::cout << "values[0]: " << this->values[0] << std::endl;
}

template <typename T>
std::vector<int>& sparse_CSR_Matrix<T>::get_row_ptr(){
    return this->row_ptr;
}

template <typename T>
std::vector<int>& sparse_CSR_Matrix<T>::get_col_idx(){
    return this->col_idx;
}

template <typename T>
MatrixType sparse_CSR_Matrix<T>::get_matrix_type() const{
    return this->matrix_type;
}

template <typename T>
std::vector<T>& sparse_CSR_Matrix<T>::get_values(){
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
    // uncomment the following during development
    if (this->development) {
        printf("WARNING Element row %d, col %d not found in sparse CSR Matrix\n", row, col);
    }
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
bool sparse_CSR_Matrix<T>::compare_to(sparse_CSR_Matrix<T>& other, std::string info) const{
    // caller_info is a string that is printed to help identify where the comparison is called from
    // this is helpful since compare to is called from a whole bunch of tests
    
    bool same = true;

    if (this->num_rows != other.get_num_rows()){
        printf("Matrices have different number of rows: this has %d the other %d for %s\n", this->num_rows, other.get_num_rows(), info.c_str());
        same = false;
    }
    if (this->num_cols != other.get_num_cols()){
        printf("Matrices have different number of cols: this has %d the other %d for %s\n", this->num_cols, other.get_num_cols(), info.c_str());
        same = false;
    }

    for (int i = 0; i < this->num_rows; i++) {
        int start = this->row_ptr[i];
        int end = this->row_ptr[i + 1];
        int other_start = other.get_row_ptr()[i];
        int other_end = other.get_row_ptr()[i + 1];
        if (end - start != other_end - other_start) {
            printf("Row %d has different number of non-zero elements for %s\n", i, info.c_str());
            same = false;
        }
        for (int j = start; j < end; j++) {
            if (this->col_idx[j] != other.get_col_idx()[j] || this->values[j] != other.get_values()[j]) {
                printf("Element at row %d, col %d is different for %s\n", i, this->col_idx[j], info.c_str());
                same = false;
            }
        }
    }
    if (same && this->development) {
        printf("Matrices are the same for %s\n", info.c_str());
    }
    return same;
}

template <typename T>
void sparse_CSR_Matrix<T>::write_to_file()const{

    std::string str_nx = std::to_string(this->nx);
    std::string str_ny = std::to_string(this->ny);
    std::string str_nz = std::to_string(this->nz);

    std::string dim_str = str_nx + "x" + str_ny + "x" + str_nz;
    
    std::string folder_path = "../../example_matrices/";
    std::string filename = folder_path + "cpp_sparse_CSR_Matrix_" + dim_str + ".txt";

    FILE * file = fopen(filename.c_str(), "w");
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    fprintf(file, "Row Pointer: ");
    for (int i = 0; i < this->num_rows + 1; i++) {
        if (i >= this->row_ptr.size()) {
            std::cerr << "Error: Accessing row_ptr out of bounds at index " << i << std::endl;
            fclose(file);
            return;
        }
        fprintf(file, "%d ", this->row_ptr[i]);
    }
    fprintf(file, "\n");

    fprintf(file, "Column Index: ");
    for (int i = 0; i < this->nnz; i++) {
        if (i >= this->col_idx.size()) {
            std::cerr << "Error: Accessing col_idx out of bounds at index " << i << std::endl;
            fclose(file);
            return;
        }
        fprintf(file, "%d ", this->col_idx[i]);
    }
    fprintf(file, "\n");

    fprintf(file, "Values: ");
    for (int i = 0; i < this->nnz; i++) {
        if (i >= this->values.size()) {
            std::cerr << "Error: Accessing values out of bounds at index " << i << std::endl;
            fclose(file);
            return;
        }
        fprintf(file, "%f ", this->values[i]);
    }
    fprintf(file, "\n");
    
    fclose(file);
}

template <typename T>
void sparse_CSR_Matrix<T>::read_from_file(std::string nx, std::string ny, std::string nz, std::string matrix_type){

    int int_nx = std::stoi(nx);
    int int_ny = std::stoi(ny);
    int int_nz = std::stoi(nz);

    this->nx = int_nx;
    this->ny = int_ny;
    this->nz = int_nz;
    this->num_rows = int_nx * int_ny * int_nz;
    this->num_cols = int_nx * int_ny * int_nz;

    // we make a new set of pointers and then reassign before we end
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> values;

    std::string dim_str = nx + "x" + ny + "x" + nz;
    
    std::string folder_path = "../../example_matrices/";
    std::string filename =  folder_path + matrix_type +"_sparse_CSR_Matrix_" + dim_str + ".txt";
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }


    std::string line;

    // Read the row pointers
    std::getline(file, line);
    std::istringstream row_ptr_stream(line.substr(std::string("Row Pointer: ").length()));
    int value;
    while (row_ptr_stream >> value) {
        row_ptr.push_back(value);
        this->nnz += 1;
    }

    // Read the column indices
    std::getline(file, line);
    std::istringstream col_idx_stream(line.substr(std::string("Column Index: ").length()));
    while (col_idx_stream >> value) {
        col_idx.push_back(value);
    }

    // Read the values
    std::getline(file, line);
    std::istringstream values_stream(line.substr(std::string("Values: ").length()));
    T val;
    while (values_stream >> val) {
        values.push_back(val);
    }

    file.close();


    this->row_ptr = row_ptr;
    this->col_idx = col_idx;
    this->values = values;
   
}

//explicit instantiation
template class sparse_CSR_Matrix<double>;

