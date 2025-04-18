#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_partial_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/utils.hpp"
#include "MatrixLib/coloring.cuh"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"

#include <vector>
#include <iostream>
#include <string>

#include <memory>

// #include <stdio.h>
template <typename T>
striped_partial_Matrix<T>::striped_partial_Matrix(Problem *p) {
    // this->nx = 0;
    // this->ny = 0;
    // this->nz = 0;
    //this->nnz = 0;
    //this->diag_index = -1;
    this->problem = p;
    this->matrix_type = MatrixType::Stencil_3D27P;
    global_int_t gnx = p->gnx;
    global_int_t gny = p->gny;
    global_int_t gnz = p->gnz;

    this->num_rows = p->nx * p->ny * p->nz;
    this->num_cols = p->nx * p->ny * p->nz;
    this->num_stripes = 27;

    global_int_t num_interior_points = (gnx - 2) * (gny - 2) * (gnz - 2);
    global_int_t num_face_points = 2 * ((gnx - 2) * (gny - 2) + (gnx - 2) * (gnz - 2) + (gny - 2) * (gnz - 2));
    global_int_t num_edge_points = 4 * ((gnx - 2) + (gny - 2) + (gnz - 2));
    global_int_t num_corner_points = 8;

    global_int_t nnz_interior = 27 * num_interior_points;
    global_int_t nnz_face = 18 * num_face_points;
    global_int_t nnz_edge = 12 * num_edge_points;
    global_int_t nnz_corner = 8 * num_corner_points;

    global_int_t nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner;

    this->nnz = nnz;


    //this->j_min_i.clear();
    //this->values.clear();
    this->j_min_i = std::vector<local_int_t>(this->num_stripes, 0);
    //this->j_min_i_d = nullptr;
    //this->values_d = nullptr;

    //this->color_pointer_d = nullptr;
    //this->color_sorted_rows_d = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op_d = nullptr;
    this->rc_d = new Halo;
    this->xc_d = new Halo;
    this->Axf_d = new Halo;
    //this->f2c_op.clear();

    CHECK_CUDA(cudaMalloc(&this->values_d, sizeof(T) * num_rows* 27));
    GenerateStripedPartialMatrix_GPU(this->problem, this->values_d);

    // fill j_min_i_d
    local_int_t neighbour_offsets [num_stripes][3] = {
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

    for (int i = 0; i < this->num_stripes; i++) {
        local_int_t off_x = neighbour_offsets[i][0];
        local_int_t off_y = neighbour_offsets[i][1];
        local_int_t off_z = neighbour_offsets[i][2];
        
        this->j_min_i[i] = off_x + off_y * p->gnx + off_z * p->gnx * p->gny;
        if (this->j_min_i[i] == 0) {
            this->diag_index = i;
        }
    }

    CHECK_CUDA(cudaMalloc(&this->j_min_i_d, this->num_stripes * sizeof(local_int_t)));
    CHECK_CUDA(cudaMemcpy(this->j_min_i_d, this->j_min_i.data(), this->num_stripes * sizeof(local_int_t), cudaMemcpyHostToDevice));
}

template<typename T>
local_int_t striped_partial_Matrix<T>::get_nx() const{
    return this->problem->nx;
}

template<typename T>
local_int_t striped_partial_Matrix<T>::get_ny() const{
    return this->problem->nx;
}

template<typename T>
local_int_t striped_partial_Matrix<T>::get_nz() const{
    return this->problem->nx;
}

template <typename T>
MatrixType striped_partial_Matrix<T>::get_matrix_type() const{
    return this->matrix_type;
}

template <typename T>
striped_partial_Matrix<T>::~striped_partial_Matrix(){

    if(this->j_min_i_d != nullptr){
        CHECK_CUDA(cudaFree(this->j_min_i_d));
        this->j_min_i_d = nullptr;
    }

    if(this->values_d != nullptr){
        CHECK_CUDA(cudaFree(this->values_d));
        this->values_d = nullptr;
    }

    /* if (this->color_pointer_d != nullptr) {
        CHECK_CUDA(cudaFree(this->color_pointer_d));
        this->color_pointer_d = nullptr;

    }
    if (this->color_sorted_rows_d != nullptr) {
        CHECK_CUDA(cudaFree(this->color_sorted_rows_d));
        this->color_sorted_rows_d = nullptr;
    } */

    if (this->f2c_op_d != nullptr) {
        CHECK_CUDA(cudaFree(this->f2c_op_d));
        this->f2c_op_d = nullptr;
    }
    if (this->rc_d != nullptr) {
        delete this->rc_d;
        this->rc_d = nullptr;
    }
    if(this->xc_d != nullptr){
        delete this->xc_d;
        this->xc_d = nullptr;
    }
    if(this->Axf_d != nullptr){
        delete this->Axf_d;
        this->Axf_d = nullptr;
    }
    
    if(this->coarse_Matrix != nullptr){
        delete this->coarse_Matrix;
        this->coarse_Matrix = nullptr;
    }
}

template <typename T>
striped_partial_Matrix<T>* striped_partial_Matrix<T>::get_coarse_Matrix(){
    return this->coarse_Matrix;
}

template <typename T>
Halo* striped_partial_Matrix<T>::get_rc_d(){
    return this->rc_d;
}

template <typename T>
Halo* striped_partial_Matrix<T>::get_xc_d(){
    return this->xc_d;
}

template <typename T>
Halo* striped_partial_Matrix<T>::get_Axf_d(){
    return this->Axf_d;
}

template <typename T>
local_int_t striped_partial_Matrix<T>::get_num_rows() const{
    return this->num_rows;
}

template <typename T>
local_int_t striped_partial_Matrix<T>::get_num_cols() const{
    return this->num_cols;
}

template <typename T>
local_int_t striped_partial_Matrix<T>::get_num_stripes() const{
    return this->num_stripes;
}

template <typename T>
global_int_t striped_partial_Matrix<T>::get_nnz() const{
    return this->nnz;
}

template <typename T>
local_int_t striped_partial_Matrix<T>::get_diag_index() const{
    return this->diag_index;
}

template <typename T>
int striped_partial_Matrix<T>::get_num_MG_pre_smooth_steps() const{
    return this->num_MG_pre_smooth_steps;
}

template <typename T>
int striped_partial_Matrix<T>::get_num_MG_post_smooth_steps() const{
    return this->num_MG_post_smooth_steps;
}

template <typename T>
local_int_t * striped_partial_Matrix<T>::get_j_min_i_d(){
    return this->j_min_i_d;
}

template <typename T>
T * striped_partial_Matrix<T>::get_values_d(){
    return this->values_d;
}

template <typename T>
local_int_t * striped_partial_Matrix<T>::get_f2c_op_d(){
    return this->f2c_op_d;
}

template <typename T>
void striped_partial_Matrix<T>::generate_f2c_operator_onGPU() {
    const local_int_t x = this->problem->nx;
    const local_int_t y = this->problem->ny;
    const local_int_t z = this->problem->nz;
    const local_int_t fine_n_rows = x *2 * y * 2 * z * 2;

    // allocate space for the device pointers
    CHECK_CUDA(cudaMalloc(&this->f2c_op_d, fine_n_rows * sizeof(local_int_t)));

    // set them to zero
    CHECK_CUDA(cudaMemset(this->f2c_op_d, 0, fine_n_rows * sizeof(local_int_t)));

    generate_partialf2c_operator(x, y, z, x*2, y*2, z*2, this->f2c_op_d);
}

template <typename T>
void striped_partial_Matrix<T>::initialize_coarse_matrix(){
    assert(this->problem->nx % 2 == 0);
    assert(this->problem->ny % 2 == 0);
    assert(this->problem->nz % 2 == 0);
    assert(this->problem->gnx % 2 == 0);
    assert(this->problem->gny % 2 == 0);
    assert(this->problem->gnz % 2 == 0);
    assert(this->coarse_Matrix == nullptr);

    local_int_t nx_f = this->problem->nx;
    local_int_t ny_f = this->problem->ny;
    local_int_t nz_f = this->problem->nz;
    //int fine_n_rows = this->nx *2 * this->ny * 2 * this->nz * 2;
    local_int_t nx_c = this->problem->nx / 2;
    local_int_t ny_c = this->problem->ny / 2;
    local_int_t nz_c = this->problem->nz / 2;

    // allocate coarse matrix
    Problem *p_c = new Problem;
    GenerateProblem(this->problem->npx, this->problem->npy, this->problem->npz, nx_c, ny_c, nz_c, this->problem->size, this->problem->rank, p_c);
    this->coarse_Matrix= new striped_partial_Matrix<T>(p_c);
    this->coarse_Matrix->generate_f2c_operator_onGPU();

    // allocate halos rc, xc, Axf and set to zero
    InitHalo(this->coarse_Matrix->get_rc_d(), p_c);
    InitHalo(this->coarse_Matrix->get_xc_d(), p_c);
    InitHalo(this->coarse_Matrix->get_Axf_d(), this->problem);

    SetHaloGlobalIndexGPU(this->coarse_Matrix->get_rc_d(), p_c);
    SetHaloGlobalIndexGPU(this->coarse_Matrix->get_xc_d(), p_c);
    SetHaloGlobalIndexGPU(this->coarse_Matrix->get_Axf_d(), this->problem);
}

template <typename T>
void striped_partial_Matrix<T>::generateMatrix_onGPU(){
    CHECK_CUDA(cudaMalloc(&this->values_d, sizeof(T)*this->num_stripes*this->num_rows));
    GenerateStripedPartialMatrix_GPU(this->problem, this->values_d);
}

// explicit template instantiation
template class striped_partial_Matrix<DataType>;