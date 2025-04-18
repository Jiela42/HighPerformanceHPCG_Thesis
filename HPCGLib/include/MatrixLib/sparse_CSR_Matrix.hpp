#ifndef SPARSE_CSR_MATRIX_HPP
#define SPARSE_CSR_MATRIX_HPP

#include "MatrixLib/striped_Matrix.hpp"
#include "MatrixLib/matrix_basics.hpp"
#include "UtilLib/utils.hpp"

#include <vector>
#include <string>
#include <cassert>


// Forward declaration of striped_Matrix
template <typename T>
class striped_Matrix;

template <typename T>
class sparse_CSR_Matrix {
public:

    bool development = false;

    sparse_CSR_Matrix();
    sparse_CSR_Matrix(int nx, int ny, int nz, local_int_t nnz, MatrixType mt, T* vals, local_int_t* row_ptr, local_int_t* col_idx);
    sparse_CSR_Matrix(int nx, int ny, int nz, local_int_t nnz, MatrixType mt, std::vector<T> vals, std::vector<local_int_t> row_ptr, std::vector<local_int_t> col_idx);
    sparse_CSR_Matrix(std::vector<std::vector<T>> dense_matrix);

    ~sparse_CSR_Matrix();


    void generateMatrix_onGPU(int nx, int ny, int nz);
    void generateMatrix_onCPU(int nx, int ny, int nz);

    void generate_f2c_operator_onGPU();
    void generate_f2c_operator_onCPU();

    
    void sanity_check_3D27P();
    void sanity_check_Matrix_on_CPU() const;

    void iterative_values();
    void random_values(int seed);

    std::vector<local_int_t>& get_row_ptr();
    std::vector<local_int_t>& get_col_idx();
    std::vector<T>& get_values();

    local_int_t* get_row_ptr_d();
    local_int_t* get_col_idx_d();
    T* get_values_d();
    
    void copy_Matrix_toGPU();
    void copy_Matrix_toCPU();
    
    void remove_Matrix_from_GPU();
    
    local_int_t get_num_rows() const;
    local_int_t get_num_cols() const;
    int get_nx() const;
    int get_ny() const;
    int get_nz() const;
    local_int_t get_nnz() const;
    int get_num_MG_pre_smooth_steps() const;
    int get_num_MG_post_smooth_steps() const;
    striped_Matrix<T>* get_Striped();
    void set_Striped(striped_Matrix<T>* A);
    sparse_CSR_Matrix<T>* get_coarse_Matrix();
    local_int_t * get_f2c_op_d();
    T * get_rc_d();
    T * get_xc_d();
    T * get_Axf_d();
    std::vector<local_int_t> get_f2c_op();
    void initialize_coarse_Matrix();

    MatrixType get_matrix_type() const;

    T get_element(local_int_t i, local_int_t j) const;
    void print() const;
    bool compare_to(sparse_CSR_Matrix<T>& other, std::string info);
    void write_to_file() const;
    void read_from_file(std::string nx, std::string ny, std::string nz, std::string matrix_type);

private:
    friend class striped_Matrix<T>;
    int nx;
    int ny;
    int nz;
    local_int_t nnz;
    local_int_t num_rows;
    local_int_t num_cols;
    std::vector<local_int_t> row_ptr;
    std::vector<local_int_t> col_idx;
    std::vector<T> values;
    local_int_t * row_ptr_d;
    local_int_t * col_idx_d;
    T * values_d;
    MatrixType matrix_type;
    
    striped_Matrix<T>* Striped;
    
    int num_MG_pre_smooth_steps;
    int num_MG_post_smooth_steps;
    sparse_CSR_Matrix<T> * coarse_Matrix;
    local_int_t * f2c_op_d;
    // since we only have the MG routines run on the GPU, we only have the coarse matrix data (except f2c_op) on the GPU
    // (they are all vectors)
    T* rc_d;
    T* xc_d;
    T* Axf_d;

    std::vector<local_int_t> f2c_op;
    
    // friend void striped_Matrix<T>::striped_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T>& A);
    void sparse_CSR_Matrix_from_striped(striped_Matrix<T> &A);
    void sparse_CSR_Matrix_from_striped_transformation(striped_Matrix<T>&);
    void sparse_CSR_Matrix_from_striped_transformation_CPU(striped_Matrix<T>&);
    void sparse_CSR_Matrix_from_striped_transformation_GPU(striped_Matrix<T>&);
};

#endif // SPARSE_CSR_MATRIX_HPP
