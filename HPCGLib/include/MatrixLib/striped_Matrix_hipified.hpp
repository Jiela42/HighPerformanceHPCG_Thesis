#ifndef STRIPED_MATRIX_HPP
#define STRIPED_MATRIX_HPP

#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"
#include "MatrixLib/matrix_basics_hipified.hpp"
#include "MatrixLib/generations_hipified.cuh"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"

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
        
        void generate_coloring();
        void generate_box_coloring();
        
        void copy_Matrix_toGPU();
        void copy_Matrix_toCPU();

        void remove_Matrix_from_GPU();
        
        local_int_t *get_color_pointer_d();
        local_int_t *get_color_sorted_rows_d();
        std::vector<local_int_t> get_color_pointer_vector();
        std::vector<local_int_t> get_color_sorted_rows_vector();
        
        local_int_t get_num_rows() const;
        local_int_t get_num_cols() const;
        int get_num_stripes() const;
        int get_nx() const;
        int get_ny() const;
        int get_nz() const;
        local_int_t get_nnz() const;
        int get_diag_index() const;
        MatrixType get_matrix_type() const;

        sparse_CSR_Matrix<T> *get_CSR();
        void set_CSR(sparse_CSR_Matrix<T> *A);

        striped_Matrix<T> *get_coarse_Matrix();
        int get_num_MG_pre_smooth_steps() const;
        int get_num_MG_post_smooth_steps() const;
        local_int_t *get_f2c_op_d();
        T * get_rc_d();
        T * get_xc_d();
        T * get_Axf_d();
        std::vector<local_int_t> get_f2c_op();

        T get_element(local_int_t i, local_int_t j) const;
        std::vector<local_int_t>& get_j_min_i();
        std::vector <T>& get_values();
        local_int_t * get_j_min_i_d();
        T * get_values_d();
        void set_num_rows(local_int_t num_rows);
        void print_COR_Format();
        void print() const;
        void Generate_striped_3D27P_Matrix_onGPU(int nx, int ny, int nz);
        void generate_f2c_operator_onGPU();
        void init_coarse_buffer();
        void initialize_coarse_Matrix();
        // void compare_to(striped_Matrix<T>& other) const;
        // void write_to_file() const;
        bool compare_to(striped_Matrix<T>& other);
        
        
        private:
        friend class sparse_CSR_Matrix<T>;
        int nx;
        int ny;
        int nz;
        local_int_t nnz;
        int diag_index;
        local_int_t num_rows;
        local_int_t num_cols;
        int num_stripes;
        std::vector<local_int_t> j_min_i;
        std::vector<T> values;
        MatrixType matrix_type;
        // friend void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_striped(striped_Matrix<T> & A);
        void striped_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T> & A);
        void striped_3D27P_Matrix_from_CSR_onCPU(sparse_CSR_Matrix<T> & A);
        void striped_3D27P_Matrix_from_CSR_onGPU(sparse_CSR_Matrix<T> & A);
        local_int_t *j_min_i_d;
        T *values_d;
        local_int_t* color_pointer_d;
        local_int_t* color_sorted_rows_d;

        sparse_CSR_Matrix<T> *CSR;

        int num_MG_pre_smooth_steps;
        int num_MG_post_smooth_steps;
        striped_Matrix<T> *coarse_Matrix;
        std::vector<local_int_t> f2c_op;
        local_int_t *f2c_op_d;
        // since we only have the MG routines run on the GPU, we only have the coarse matrix data (except f2c_op) on the GPU
        // (they are all vectors)
        T* rc_d;
        T* xc_d;
        T* Axf_d;

};

#endif // STRIPED_MATRIX_HPP