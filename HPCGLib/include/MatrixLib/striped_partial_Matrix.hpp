#ifndef STRIPED_PARTIAL_MATRIX_HPP
#define STRIPED_PARTIAL_MATRIX_HPP

#include "MatrixLib/matrix_basics.hpp"
#include "MatrixLib/generations.cuh"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "UtilLib/utils.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cassert>

template <typename T>
class striped_partial_Matrix {
    public:
        striped_partial_Matrix(Problem *problem);
        ~striped_partial_Matrix();
        
        // void generate_coloring();
        
        //void copy_Matrix_toGPU();
        //void copy_Matrix_toCPU();

        //void remove_Matrix_from_GPU();
        
        //int *get_color_pointer_d();
        //int *get_color_sorted_rows_d();
        //std::vector<int> get_color_pointer_vector();
        //std::vector<int> get_color_sorted_rows_vector();
        
        local_int_t get_num_rows() const;
        local_int_t get_num_cols() const;
        local_int_t get_num_stripes() const;
        local_int_t get_nx() const;
        local_int_t get_ny() const;
        local_int_t get_nz() const;
        global_int_t get_nnz() const;
        local_int_t get_diag_index() const;
        //int get_nnz() const;
        MatrixType get_matrix_type() const;

        //sparse_CSR_Matrix<T> *get_CSR();
        //void set_CSR(sparse_CSR_Matrix<T> *A);

        striped_partial_Matrix<T> *get_coarse_Matrix();
        int get_num_MG_pre_smooth_steps() const;
        int get_num_MG_post_smooth_steps() const;
        local_int_t* get_f2c_op_d();
        Halo* get_rc_d();
        Halo* get_xc_d();
        Halo* get_Axf_d();
        std::vector<local_int_t> get_f2c_op();

        // T get_element(int i, int j) const;
        //std::vector <int>& get_j_min_i();
        //std::vector <T>& get_values();
        local_int_t * get_j_min_i_d();
        T * get_values_d();
        //void set_num_rows(int num_rows);
        //void print_COR_Format();
        //void print() const;
        // void compare_to(striped_Matrix<T>& other) const;
        // void write_to_file() const;

        void initialize_coarse_matrix();
        void generateMatrix_onGPU();     
        void generate_f2c_operator_onGPU();   
        Problem *get_problem() {return problem;};
        
        private:
        //friend class sparse_CSR_Matrix<T>;
        Problem *problem;
        local_int_t nx;
        local_int_t ny;
        local_int_t nz;
        global_int_t nnz;
        local_int_t diag_index;
        local_int_t num_rows;
        local_int_t num_cols;
        int num_stripes;
        std::vector<local_int_t> j_min_i;
        // std::vector<T> values;
        MatrixType matrix_type;
        // friend void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_striped(striped_Matrix<T> & A);
        // void striped_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T> & A);
        // void striped_3D27P_Matrix_from_CSR_onCPU(sparse_CSR_Matrix<T> & A);
        // void striped_3D27P_Matrix_from_CSR_onGPU(sparse_CSR_Matrix<T> & A);
        local_int_t *j_min_i_d;
        T *values_d;
        // int* color_pointer_d;
        // int* color_sorted_rows_d;

        // sparse_CSR_Matrix<T> *CSR;

        int num_MG_pre_smooth_steps;
        int num_MG_post_smooth_steps;
        striped_partial_Matrix<T> *coarse_Matrix;
        // std::vector<int> f2c_op;
        local_int_t *f2c_op_d;
        // since we only have the MG routines run on the GPU, we only have the coarse matrix data (except f2c_op) on the GPU
        // (they are all vectors)
        Halo* rc_d;
        Halo* xc_d;
        Halo* Axf_d;

};



#endif // STRIPED_PARTIAL_MATRIX_HPP
