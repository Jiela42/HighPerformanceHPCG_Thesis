#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/generations.cuh"
#include "UtilLib/cuda_utils.hpp"
#include "UtilLib/hpcg_mpi_utils.cuh"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>
// #include <memory>


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

    this->Striped = nullptr;
    // std::cout << "sparse_CSR_Matrix created, setting device ptrs to null" << std::endl;
    this->row_ptr_d = nullptr;
    this->col_idx_d = nullptr;
    this->values_d = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->f2c_op.clear();
    this->f2c_op_d = nullptr;
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
    this->coarse_Matrix = nullptr;
    this->Striped = nullptr;
}

template <typename T>
sparse_CSR_Matrix<T>::~sparse_CSR_Matrix(){
    // std::cout << "Deleting sparse_CSR_Matrix" << std::endl;

    if (this->row_ptr_d != nullptr) {
        // std::cout << row_ptr_d << std::endl;
        // std::cout << "Freeing row_ptr_d" << std::endl;
        CHECK_CUDA(cudaFree(this->row_ptr_d));
        // std::cout << "row_ptr_d freed" << std::endl;
        this->row_ptr_d = nullptr;
    }
    if (this->col_idx_d != nullptr) {
        CHECK_CUDA(cudaFree(this->col_idx_d));
        this->col_idx_d = nullptr;
    }
    if (this->values_d != nullptr) {
        CHECK_CUDA(cudaFree(this->values_d));
        this->values_d = nullptr;
    }
    if(this->f2c_op_d != nullptr){
        CHECK_CUDA(cudaFree(this->f2c_op_d));
        this->f2c_op_d = nullptr;
    }
    if(this->rc_d != nullptr){
        CHECK_CUDA(cudaFree(this->rc_d));
        this->rc_d = nullptr;
    }
    if(this->xc_d != nullptr){
        CHECK_CUDA(cudaFree(this->xc_d));
        this->xc_d = nullptr;
    }
    if(this->Axf_d != nullptr){
        CHECK_CUDA(cudaFree(this->Axf_d));
        this->Axf_d = nullptr;
    }
    
    if(this->Striped != nullptr){

        // this deletion causes a deadlock (because CSR points to Striped and vice versa)
        striped_Matrix<T> *temp = this->Striped;
        this->Striped = nullptr;
        // we also have to set this matrix to null in our sibling matrix
        temp->CSR = nullptr;
        // when we delete the matrix, we also delete any coarse matrices, so we have to set that to null_ptr as well
        temp->coarse_Matrix = nullptr;
        delete temp;
    }
    if(this->coarse_Matrix != nullptr){
        delete this->coarse_Matrix;
        this->coarse_Matrix = nullptr;
    }
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

    this->row_ptr_d = nullptr;
    this->col_idx_d = nullptr;
    this->values_d = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op_d = nullptr;
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
    this->f2c_op.clear();
    this->Striped = nullptr;

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

    this->row_ptr_d = nullptr;
    this->col_idx_d = nullptr;
    this->values_d = nullptr;

    this->Striped = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op_d = nullptr;
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
}

template <typename T>
sparse_CSR_Matrix<T>::sparse_CSR_Matrix(std::vector<std::vector<T>> dense_matrix){
    
    // check that we have two dimensions
    assert(dense_matrix.size() > 0);

    int num_rows = dense_matrix.size();
    int num_cols = dense_matrix[0].size();

    for(int i = 0; i < num_rows; i++){
        assert(dense_matrix[i].size() == num_cols);
    }

    int nnz = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> values;

    row_ptr.push_back(0);

    // now we read the data into the sparse matrix
    for (int i = 0; i< num_rows; i++){
        for (int j = 0; j < num_cols; j++){
            T val = dense_matrix[i][j];
            if(val != 0){
                values.push_back(val);
                col_idx.push_back(j);
                nnz++;
            }
        }
        row_ptr.push_back(nnz);
    }

    this->matrix_type = MatrixType::UNKNOWN;
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->nnz = nnz;
    this->row_ptr = row_ptr;
    this->col_idx = col_idx;
    this->values = values;

    this->row_ptr_d = nullptr;
    this->col_idx_d = nullptr;
    this->values_d = nullptr;

    this->Striped = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op.clear();
    this->f2c_op_d = nullptr;
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;

}

template <typename T>
void sparse_CSR_Matrix<T>::set_Striped(striped_Matrix<T>* A){
    this->Striped = A;
}

template <typename T>
striped_Matrix<T>* sparse_CSR_Matrix<T>::get_Striped(){
    if(this->Striped == nullptr){
        striped_Matrix <T>* A = new striped_Matrix<T>();
        A->striped_Matrix_from_sparse_CSR(*this);
    }
    return this->Striped;
}

template <typename T>
void sparse_CSR_Matrix<T>::generateMatrix_onCPU(int nx, int ny, int nz){
    // currently this only supports 27pt 3D stencils
    this->matrix_type = MatrixType::Stencil_3D27P;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->num_rows = nx * ny * nz;
    this->num_cols = nx * ny * nz;

    int num_rows = nx * ny * nz;
    int num_cols = nx * ny * nz;

    int nnz = 0;

    std::vector<int> nnz_per_row(num_rows);

    std::vector<std::vector<int>> col_idx_per_row(num_rows);
    std::vector<std::vector<double>> values_per_row(num_rows);

    for(int ix = 0; ix < nx; ix++){
        for(int iy = 0; iy < ny; iy++){
            for(int iz = 0; iz < nz; iz++){

                int i = ix + nx * iy + nx * ny * iz;
                int nnz_i = 0;

                for (int sz = -1; sz < 2; sz++){
                    if(iz + sz > -1 && iz + sz < nz){
                        for(int sy = -1; sy < 2; sy++){
                            if(iy + sy > -1 && iy + sy < ny){
                                for(int sx = -1; sx < 2; sx++){
                                    if(ix + sx > -1 && ix + sx < nx){
                                        int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                        if(i == j){
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(26.0);
                                        } else {
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(-1.0);
                                        }
                                            nnz_i++;
                                            nnz++;
                                    }
                                }
                            }
                        }
                    }
                }
                nnz_per_row[i] = nnz_i;
            }
        }
    }

    this->row_ptr = std::vector<int>(num_rows + 1, 0);
    this->col_idx = std::vector<int>();
    this->values = std::vector<T>();

    for (int i = 0; i < num_rows; i++){
        this->row_ptr[i + 1] = this->row_ptr[i] + nnz_per_row[i];

        for (int j = 0; j < nnz_per_row[i]; j++){
            this->col_idx.push_back(col_idx_per_row[i][j]);
            this->values.push_back(values_per_row[i][j]);
        }
    }

    this->nnz = nnz;

    this->row_ptr_d = nullptr;
    this->col_idx_d = nullptr;
    this->values_d = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op_d = nullptr;
    this->f2c_op.clear();
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
}

template <typename T>
void sparse_CSR_Matrix<T>::generatePartialMatrix_onCPU(Problem *problem){
    // currently this only supports 27pt 3D stencils
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    global_int_t gnx = problem->gnx;
    global_int_t gny = problem->gny;
    global_int_t gnz = problem->gnz;
    global_int_t gi0 = problem->gi0;
    
    
    this->matrix_type = MatrixType::Stencil_3D27P;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->num_rows = nx * ny * nz;
    this->num_cols = nx * ny * nz;
    
    int num_rows = nx * ny * nz;
    int num_cols = nx * ny * nz;
    
    int nnz = 0;
    
    std::vector<int> nnz_per_row(num_rows);
    
    std::vector<std::vector<int>> col_idx_per_row(num_rows);
    std::vector<std::vector<double>> values_per_row(num_rows);
    
    global_int_t gx = problem->gx0;
    global_int_t gy = problem->gy0;
    global_int_t gz = problem->gz0;
    
    for(int ix = 0; ix < nx; ix++){
        for(int iy = 0; iy < ny; iy++){
            for(int iz = 0; iz < nz; iz++){

                int i = ix + nx * iy + nx * ny * iz;

                //convert local i to global i
                int local_i_x = i % nx;
                int local_i_y = (i % (nx * ny)) / nx;
                int local_i_z = i / (nx * ny);
                global_int_t gi = gi0 + local_i_x + local_i_y * gnx + local_i_z * (gnx * gny);

                int nnz_i = 0;

                for (int sz = -1; sz < 2; sz++){
                    if(gz + sz > -1 && gz + sz < gnz){
                        for(int sy = -1; sy < 2; sy++){
                            if(gy + sy > -1 && gy + sy < gny){
                                for(int sx = -1; sx < 2; sx++){
                                    if(gx + sx > -1 && gx + sx < gnx){
                                        int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                        //convert local j to global j
                                        int local_j_x = j % nx;
                                        int local_j_y = (j % (nx * ny)) / nx;
                                        int local_j_z = j / (nx * ny);
                                        global_int_t gj = gi0 + local_j_x + local_j_y * gnx + local_j_z * (gnx * gny);

                                        if(gi == gj){
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(26.0);
                                        } else {
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(-1.0);
                                        }
                                            nnz_i++;
                                            nnz++;
                                    }
                                }
                            }
                        }
                    }
                }
                nnz_per_row[i] = nnz_i;
                gz++;
            }
            gy++;
        }
        gx++;
    }

    this->row_ptr = std::vector<int>(num_rows + 1, 0);
    this->col_idx = std::vector<int>(nnz, 0);
    this->values = std::vector<T>(nnz, 0);

    for (int i = 0; i < num_rows; i++){
        this->row_ptr[i + 1] = this->row_ptr[i] + nnz_per_row[i];

        for (int j = 0; j < nnz_per_row[i]; j++){
            this->col_idx.push_back(col_idx_per_row[i][j]);
            this->values.push_back(values_per_row[i][j]);
        }
    }

    this->nnz = nnz;

    this->row_ptr_d = nullptr;
    this->col_idx_d = nullptr;
    this->values_d = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op_d = nullptr;
    this->f2c_op.clear();
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
}

template<typename T>
void sparse_CSR_Matrix<T>::generateMatrix_onGPU(int nx, int ny, int nz)
{
    // currently this only supports 27pt 3D stencils
    assert(nx > 2 and ny > 2 and nz > 2);
    this->matrix_type = MatrixType::Stencil_3D27P;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;

    int num_nodes = nx * ny * nz;

    this->num_rows = num_nodes;
    this->num_cols = num_nodes;

    int num_interior_points = (nx - 2) * (ny - 2) * (nz - 2);
    int num_face_points = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2));
    int num_edge_points = 4 * ((nx - 2) + (ny - 2) + (nz - 2));
    int num_corner_points = 8;

    int nnz_interior = 27 * num_interior_points;
    int nnz_face = 18 * num_face_points;
    int nnz_edge = 12 * num_edge_points;
    int nnz_corner = 8 * num_corner_points;

    int nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner;

    this->nnz = nnz;

    // allocate space for the device pointers
    CHECK_CUDA(cudaMalloc(&this->row_ptr_d, (num_nodes + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&this->col_idx_d, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&this->values_d, nnz * sizeof(T)));

    // set them to zero
    // std::cout << "row_ptr_d: " << row_ptr_d << std::endl;
    CHECK_CUDA(cudaMemset(this->row_ptr_d, 0, (num_nodes + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(this->col_idx_d, 0, nnz * sizeof(int)));
    CHECK_CUDA(cudaMemset(this->values_d, 0, nnz * sizeof(T)));

    // std::cout << "generating CSR Matrix on GPU" << std::endl;

    // now we generate the matrix
    generateHPCGMatrix(nx, ny, nz, this->row_ptr_d, this->col_idx_d, this->values_d);

}

template <typename T>
void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_striped(striped_Matrix<T> &A){

    // std::cout << "sparse_CSR_Matrix_from_striped" << std::endl;

    if(A.get_matrix_type() == MatrixType::Stencil_3D27P){
        assert(A.get_num_stripes() == 27);
        assert(A.get_num_rows() == A.get_num_cols());
        assert(A.get_num_rows() == A.get_nx() * A.get_ny() * A.get_nz());
    }

    this->matrix_type = A.get_matrix_type();
    this->nx = A.get_nx();
    this->ny = A.get_ny();
    this->nz = A.get_nz();
    this->nnz = A.get_nnz();
    this->num_rows = A.get_num_rows();
    this->num_cols = A.get_num_cols();
    this->num_MG_pre_smooth_steps = A.get_num_MG_pre_smooth_steps();
    this->num_MG_post_smooth_steps = A.get_num_MG_post_smooth_steps();

    this->Striped = &A;
    A.set_CSR(this);

    assert(A.get_num_stripes() == A.get_j_min_i().size());

    // check if A is on the GPU
    if(A.get_values_d() != nullptr and A.get_j_min_i_d() != nullptr){
        // std::cout << "A is on the GPU" << std::endl;
        this->sparse_CSR_Matrix_from_striped_transformation_GPU(A);
    }
    else{
        // std::cout << "A is on the CPU" << std::endl;
        this->sparse_CSR_Matrix_from_striped_transformation_CPU(A);
    }
}

template <typename T>
void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_striped_transformation_CPU(striped_Matrix<T> & A){
    
    this->Striped = &A;
    A.set_CSR(this);
    
    this->row_ptr = std::vector<int>(this->num_rows + 1, 0);
    this->col_idx = std::vector<int>(this->nnz, 0);
    this->values = std::vector<T>(this->nnz, 0);

    // for(int i =0; i< A.get_num_stripes(); i++){
    //     std::cout << A.get_values()[i] << std::endl;
    // }

    std::vector<T> striped_vals = A.get_values();

    // for(int i=0; i< A.get_num_stripes(); i++){
    //     std::cout << "striped_vals: " << striped_vals[i] << std::endl;
    // }
    int elem_count = 0;

    for(int i = 0; i < this->num_rows; i++){
        int nnz_i = 0;
        for(int stripe_j = 0; stripe_j < A.get_num_stripes(); stripe_j++){
            int j = A.get_j_min_i()[stripe_j] + i;
            double val = striped_vals[i*A.get_num_stripes() + stripe_j];
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

    if(A.get_f2c_op().size() > 0){
        this->f2c_op = A.get_f2c_op();
    } else{
        this->f2c_op.clear();
    }

    if(A.get_coarse_Matrix() != nullptr){
        this->coarse_Matrix = new sparse_CSR_Matrix<T>();
        this->coarse_Matrix->sparse_CSR_Matrix_from_striped(*(A.get_coarse_Matrix()));
    }
}

template <typename T>
void sparse_CSR_Matrix<T>::sparse_CSR_Matrix_from_striped_transformation_GPU(striped_Matrix<T> & A){

    // allocate space for the device pointers
    CHECK_CUDA(cudaMalloc(&this->row_ptr_d, (this->num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&this->col_idx_d, this->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&this->values_d, this->nnz * sizeof(T)));

    // set them to zero
    CHECK_CUDA(cudaMemset(this->row_ptr_d, 0, (this->num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(this->col_idx_d, 0, this->nnz * sizeof(int)));
    CHECK_CUDA(cudaMemset(this->values_d, 0, this->nnz * sizeof(T)));

    int new_nnz = generate_CSR_from_Striped(
        this->num_rows, A.get_num_stripes(),
        A.get_j_min_i_d(), A.get_values_d(),
        this->row_ptr_d, this->col_idx_d, this->values_d
    );
    
    assert(new_nnz == this->nnz);
    assert(new_nnz == A.get_nnz());

    if(A.get_f2c_op_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->f2c_op_d, this->num_rows * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(this->f2c_op_d, A.get_f2c_op_d(), this->num_rows * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    if(A.get_rc_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->rc_d, this->num_rows * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(this->rc_d, A.get_rc_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    if(A.get_xc_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->xc_d, this->num_rows * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(this->xc_d, A.get_xc_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    if(A.get_Axf_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->Axf_d, this->num_rows * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(this->Axf_d, A.get_Axf_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    if(A.get_coarse_Matrix() != nullptr){
        this->coarse_Matrix = new sparse_CSR_Matrix<T>();
        this->coarse_Matrix->sparse_CSR_Matrix_from_striped(*(A.get_coarse_Matrix()));
    }

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
void sparse_CSR_Matrix<T>::sanity_check_Matrix_on_CPU() const {
    assert(this->num_rows == this->row_ptr.size() - 1);
    assert(this->nnz == this->col_idx.size());
    assert(this->nnz == this->values.size());
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

    // these assume that the matrix is on the CPU, so if they are not, we need to copy them to the CPU
    if(this->row_ptr_d != nullptr or this->col_idx_d != nullptr or this->values_d != nullptr){
        this->copy_Matrix_toCPU();
    }

    return this->row_ptr;
}

template <typename T>
std::vector<int>& sparse_CSR_Matrix<T>::get_col_idx(){
    // these assume that the matrix is on the CPU, so if they are not, we need to copy them to the CPU
    if(this->row_ptr_d != nullptr or this->col_idx_d != nullptr or this->values_d != nullptr){
        this->copy_Matrix_toCPU();
    }
    return this->col_idx;
}

template <typename T>
std::vector<T>& sparse_CSR_Matrix<T>::get_values(){
    // these assume that the matrix is on the CPU, so if they are not, we need to copy them to the CPU
    if(this->row_ptr_d != nullptr or this->col_idx_d != nullptr or this->values_d != nullptr){
        this->copy_Matrix_toCPU();
    }
    return this->values;
}

template <typename T>
int* sparse_CSR_Matrix<T>::get_row_ptr_d(){

    // these assume that the matrix is on the GPU, so if they are not, we need to copy them to the GPU
    // if(this->row_ptr_d == nullptr or this->col_idx_d == nullptr or this->values_d == nullptr){
        // this->copy_Matrix_toGPU();
    // }
    return this->row_ptr_d;
}

template <typename T>
int* sparse_CSR_Matrix<T>::get_col_idx_d(){
    // these assume that the matrix is on the GPU, so if they are not, we need to copy them to the GPU
    // if(this->row_ptr_d == nullptr or this->col_idx_d == nullptr or this->values_d == nullptr){
    //     this->copy_Matrix_toGPU();
    // }
    return this->col_idx_d;
}

template <typename T>
T* sparse_CSR_Matrix<T>::get_values_d(){
    // these assume that the matrix is on the GPU, so if they are not, we need to copy them to the GPU
    // if(this->row_ptr_d == nullptr or this->col_idx_d == nullptr or this->values_d == nullptr){
    //     this->copy_Matrix_toGPU();
    // }
    return this->values_d;
}

template <typename T>
T* sparse_CSR_Matrix<T>::get_rc_d(){
    return this->rc_d;
}

template <typename T>
T* sparse_CSR_Matrix<T>::get_xc_d(){
    return this->xc_d;
}

template <typename T>
T* sparse_CSR_Matrix<T>::get_Axf_d(){
    return this->Axf_d;
}

template <typename T>
void sparse_CSR_Matrix<T>::copy_Matrix_toGPU(){
    
    std::cout << "Warning: copying a sparse CSR Matrix to the GPU" << std::endl;

    // clear the data from the GPU
    this->remove_Matrix_from_GPU();

    // allocate space for the device pointers
    CHECK_CUDA(cudaMalloc(&this->row_ptr_d, (this->num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&this->col_idx_d, this->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&this->values_d, this->nnz * sizeof(T)));
    
    // copy the data to the GPU
    CHECK_CUDA(cudaMemcpy(this->row_ptr_d, this->row_ptr.data(), (this->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(this->col_idx_d, this->col_idx.data(), this->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(this->values_d, this->values.data(), this->nnz * sizeof(T), cudaMemcpyHostToDevice));
    
    // not every matrix has a f2c operator
    if(not this->f2c_op.empty()){
        CHECK_CUDA(cudaMalloc(&this->f2c_op_d, this->num_rows * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(this->f2c_op_d, this->f2c_op.data(), this->num_rows * sizeof(int), cudaMemcpyHostToDevice));
    }

    // we also need to copy the coarse matrix if it exists
    if(this->coarse_Matrix != nullptr){
        this->coarse_Matrix->copy_Matrix_toGPU();
    }
}

template <typename T>
void sparse_CSR_Matrix<T>::copy_Matrix_toCPU(){

    // check that the matrix is on the GPU
    assert(this->row_ptr_d != nullptr);
    assert(this->col_idx_d != nullptr);
    assert(this->values_d != nullptr);
    
    // we do not assert this, because not every Matrix has a fine to coarse operator
    if(this->f2c_op_d != nullptr){
        this->f2c_op.resize(this->num_rows);
        CHECK_CUDA(cudaMemcpy(this->f2c_op.data(), this->f2c_op_d, this->num_rows * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Resize the vectors to the appropriate size
    this->row_ptr.resize(this->num_rows + 1);
    this->col_idx.resize(this->nnz);
    this->values.resize(this->nnz);
    
    // copy the data to the CPU
    CHECK_CUDA(cudaMemcpy(this->row_ptr.data(), this->row_ptr_d, (this->num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(this->col_idx.data(), this->col_idx_d, this->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(this->values.data(), this->values_d, this->nnz * sizeof(T), cudaMemcpyDeviceToHost));
    

    // we also need to copy the coarse matrix if it exists
    if(this->coarse_Matrix != nullptr){
        // std::cout << "copying coarse matrix to CPU" << std::endl;
        this->coarse_Matrix->copy_Matrix_toCPU();
    }
}

// these functions should not be called lightly by a user.
// we expose them since we need them in case of a mess up in the striped matrix
// but we don't really want anyone to call them (except for the devs)
template <typename T>
void sparse_CSR_Matrix<T>::remove_Matrix_from_GPU(){

    if(this->row_ptr_d != nullptr or this->col_idx_d != nullptr or this->values_d != nullptr){
    std::cerr << "WARNING: remove_Matrix_from_GPU called" << std::endl;
    std::cerr << "If the matrix did not contain relevant data, this is okay" << std::endl;
    }
    if (this->row_ptr_d != nullptr) {
        CHECK_CUDA(cudaFree(this->row_ptr_d));
        this->row_ptr_d = nullptr;
    }
    if (this->col_idx_d != nullptr) {
        CHECK_CUDA(cudaFree(this->col_idx_d));
        this->col_idx_d = nullptr;
    }
    if (this->values_d != nullptr) {
        CHECK_CUDA(cudaFree(this->values_d));
        this->values_d = nullptr;
    }
    if(this->f2c_op_d != nullptr){
        CHECK_CUDA(cudaFree(this->f2c_op_d));
        this->f2c_op_d = nullptr;
    }
    if(this->rc_d != nullptr){
        CHECK_CUDA(cudaFree(this->rc_d));
        this->rc_d = nullptr;
    }
    if(this->xc_d != nullptr){
        CHECK_CUDA(cudaFree(this->xc_d));
        this->xc_d = nullptr;
    }
    if(this->Axf_d != nullptr){
        CHECK_CUDA(cudaFree(this->Axf_d));
        this->Axf_d = nullptr;
    }
}


template <typename T>
void sparse_CSR_Matrix<T>::generate_f2c_operator_onGPU(){

    int fine_n_rows = this->nx *2 * this->ny * 2 * this->nz * 2;

    // allocate space for the device pointers
    CHECK_CUDA(cudaMalloc(&this->f2c_op_d, fine_n_rows * sizeof(int)));

    // set them to zero
    CHECK_CUDA(cudaMemset(this->f2c_op_d, 0, fine_n_rows * sizeof(int)));

    generate_f2c_operator(
        this->nx, this->ny, this->nz,
        (this->nx * 2), (this->ny * 2), (this->nz * 2),
        this->f2c_op_d);
}

template <typename T>
void sparse_CSR_Matrix<T>::generate_f2c_operator_onCPU(){
    
    int nxf = this->nx * 2;
    int nyf = this->ny * 2;
    int nzf = this->nz * 2;
    
    int nxc = this->nx;
    int nyc = this->ny;
    int nzc = this->nz;

    this->f2c_op = std::vector<int>(nxf*nyf*nzf, 0);

    for (int izc=0; izc<nzc; ++izc) {
        int izf = 2*izc;
        for (int iyc=0; iyc<nyc; ++iyc) {
            int iyf = 2*iyc;
            for (int ixc=0; ixc<nxc; ++ixc) {
                int ixf = 2*ixc;
                int currentCoarseRow = izc*nxc*nyc+iyc*nxc+ixc;
                int currentFineRow = izf*nxf*nyf+iyf*nxf+ixf;
                this->f2c_op[currentCoarseRow] = currentFineRow;
            } // end iy loop
        } // end even iz if statement
    } // end iz loop

    // std::cout << "from generate_f2c on cpu f2c_op size: " << this->f2c_op.size() << std::endl;

}

template <typename T>
void sparse_CSR_Matrix<T>::initialize_coarse_Matrix(){
    assert(this->nx % 2 == 0);
    assert(this->ny % 2 == 0);
    assert(this->nz % 2 == 0);
    assert(this->coarse_Matrix == nullptr);
    // currently we only support 3D 27pt stencils
    assert(this->matrix_type == MatrixType::Stencil_3D27P);

    int nx_c = this->nx / 2;
    int ny_c = this->ny / 2;
    int nz_c = this->nz / 2;

    int nx_f = this->nx;
    int ny_f = this->ny;
    int nz_f = this->nz;

    // std::cout << "Initializing coarse matrix with nx: " << nx_c << " ny: " << ny_c << " nz: " << nz_c << std::endl;

    this->coarse_Matrix = new sparse_CSR_Matrix<T>();

    if(this->row_ptr_d != nullptr and this->col_idx_d != nullptr and this->values_d != nullptr){
        // in case the matrix is on the GPU, we generate the new one on the GPU as well
        // std::cout << "generating coarse matrix on the GPU" << std::endl;
        this->coarse_Matrix->generateMatrix_onGPU(nx_c, ny_c, nz_c);
        this->coarse_Matrix->generate_f2c_operator_onGPU();
        // initialize the pointers
        CHECK_CUDA(cudaMalloc(&this->coarse_Matrix->rc_d, nx_c * ny_c * nz_c * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&this->coarse_Matrix->xc_d, nx_c * ny_c * nz_c * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&this->coarse_Matrix->Axf_d, nx_f * ny_f * nz_f * sizeof(T)));

        // set them to zero
        CHECK_CUDA(cudaMemset(this->coarse_Matrix->rc_d, 0, nx_c * ny_c * nz_c * sizeof(T)));
        CHECK_CUDA(cudaMemset(this->coarse_Matrix->xc_d, 0, nx_c * ny_c * nz_c * sizeof(T)));
        CHECK_CUDA(cudaMemset(this->coarse_Matrix->Axf_d, 0, nx_f * ny_f * nz_f * sizeof(T)));

        // std::cout << "coarse matrix generated on the GPU" << std::endl;
    }
    if(not this->row_ptr.empty() and not this->col_idx.empty() and not this->values.empty()){
        // in case the matrix is on the CPU, we generate the new one on the CPU as well
        // if we already have the data on the GPU, we just copy it to the CPU
        if(this->coarse_Matrix->get_row_ptr_d() != nullptr or this->coarse_Matrix->get_col_idx_d() != nullptr or this->coarse_Matrix->get_values_d() != nullptr){
            this->coarse_Matrix->copy_Matrix_toCPU();
        } else{
            // std::cout << "generating coarse matrix on the CPU" << std::endl;
            this->coarse_Matrix->generateMatrix_onCPU(nx_c, ny_c, nz_c);
            this->coarse_Matrix->generate_f2c_operator_onCPU();
            // std::cout << "coarse matrix generated on the CPU" << std::endl;
        }      
    }

    // we need to make sure that our striped sibling has the same coarse matrix, too
    if(this->Striped != nullptr){
        // unfortunately we don't have a nice initialization (yet)
        // so this hacky code will have to do for now
        this->Striped->coarse_Matrix = new striped_Matrix<T>();
        // this will also set the sibling pointers. So yay
        this->Striped->coarse_Matrix->striped_Matrix_from_sparse_CSR(*(this->coarse_Matrix));
    }
}

template <typename T>
MatrixType sparse_CSR_Matrix<T>::get_matrix_type() const{
    return this->matrix_type;
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
int sparse_CSR_Matrix<T>::get_num_MG_pre_smooth_steps() const{
    return this->num_MG_pre_smooth_steps;
}

template <typename T>
int sparse_CSR_Matrix<T>::get_num_MG_post_smooth_steps() const{
    return this->num_MG_post_smooth_steps;
}

template <typename T>
sparse_CSR_Matrix<T>* sparse_CSR_Matrix<T>::get_coarse_Matrix(){
    return this->coarse_Matrix;
}

template <typename T>
int* sparse_CSR_Matrix<T>::get_f2c_op_d(){
    return this->f2c_op_d;
}

template <typename T>
std::vector<int> sparse_CSR_Matrix<T>::get_f2c_op(){
    // std::cout << "from sparse_matrix size of f2c_op: " << this->f2c_op.size() << std::endl;
    return this->f2c_op;
}

template <typename T>
T sparse_CSR_Matrix<T>::get_element(int row, int col) const{
    // this currently only works if the matrix is on the CPU
    this->sanity_check_Matrix_on_CPU();
    
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
    // this currently only works if the matrix is on the CPU
    this->sanity_check_Matrix_on_CPU();

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
bool sparse_CSR_Matrix<T>::compare_to(sparse_CSR_Matrix<T>& other, std::string info){
    // caller_info is a string that is printed to help identify where the comparison is called from
    // this is helpful since compare to is called from a whole bunch of tests
    
    // since we only compare on the CPU, we need to make sure that both matrices are on the CPU
    if(this->row_ptr_d != nullptr or this->col_idx_d != nullptr or this->values_d != nullptr){
        // std::cout << "Copying this matrix to CPU" << std::endl;
        this->copy_Matrix_toCPU();
        // std::cout << "Finished copying this matrix to CPU" << std::endl;
    }

    if(other.get_row_ptr_d() != nullptr or other.get_col_idx_d() != nullptr or other.get_values_d() != nullptr){
        // std::cout << "Copying other matrix to CPU" << std::endl;
        other.copy_Matrix_toCPU();
        // std::cout << "Finished copying other matrix to CPU" << std::endl;
    }

    bool same = true;

    if (this->num_rows != other.get_num_rows()){
        printf("Matrices have different number of rows: this has %d the other %d for %s\n", this->num_rows, other.get_num_rows(), info.c_str());
        same = false;
    }
    if (this->num_cols != other.get_num_cols()){
        printf("Matrices have different number of cols: this has %d the other %d for %s\n", this->num_cols, other.get_num_cols(), info.c_str());
        same = false;
    }

    std::cout << "only compared meta data" << std::endl;
    
    if (same && this->development) {
        std::cout << "Comparing row by row" << std::endl;
        for (int i = 0; i < this->num_rows; i++) {
            int start = this->row_ptr[i];
            int end = this->row_ptr[i + 1];
            int other_start = other.get_row_ptr()[i];
            int other_end = other.get_row_ptr()[i + 1];
            std::cout << "compared row " << i << std::endl;
            if (end - start != other_end - other_start) {
                printf("Row %d has different number of non-zero elements for %s\n", i, info.c_str());
                // printf("This has %d, other has %d\n", end - start, other_end - other_start);
                same = false;
            }
            for (int j = start; j < end; j++) {
                if (this->col_idx[j] != other.get_col_idx()[j] || this->values[j] != other.get_values()[j]) {
                    printf("Element at row %d, col %d is different for %s\n", i, this->col_idx[j], info.c_str());
                    same = false;
                }
            }
        }

        printf("Matrices are the same for %s\n", info.c_str());
    }

    // we also compare the f2c operator
    if(this->f2c_op.empty() and not other.get_f2c_op().empty() or
        not this->f2c_op.empty() and other.get_f2c_op().empty()){
            printf("One matrix has a f2c operator and the other does not for %s\n", info.c_str());
            same = false;
    } else if(not this->f2c_op.empty() and not other.get_f2c_op().empty()){
        std::vector<int> other_f2c = other.get_f2c_op();

        if(not other_f2c.size() == this->f2c_op.size()){
            printf("f2c operators have different sizes for %s\n", info.c_str());
            same = false;
        }

        for(int i = 0; i < other_f2c.size(); i++){
            if(other_f2c[i] != this->f2c_op[i]){
                printf("f2c operators are different for %s\n", info.c_str());
                same = false;
            }
        }
    }


    if(this->coarse_Matrix != nullptr and other.get_coarse_Matrix() != nullptr){
        std::cout << "Comparing coarse matrices" << std::endl;
        same = same and this->coarse_Matrix->compare_to(*other.get_coarse_Matrix(), info);
    }
    else if(this->coarse_Matrix != nullptr or other.get_coarse_Matrix() != nullptr){
        printf("One matrix has a coarse matrix and the other does not for %s\n", info.c_str());
        same = false;
    }

    // if(this->coarse_Matrix != nullptr){
    //     std::cout << "Coarse matrix is not null" << std::endl;
    // }

    // std::cout << "Returning same" << std::endl;

    return same;
}

template <typename T>
void sparse_CSR_Matrix<T>::write_to_file()const{
    // this currently only works if the matrix is on the CPU
    this->sanity_check_Matrix_on_CPU();

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
    // this currently only works if the matrix is on the CPU
    this->sanity_check_Matrix_on_CPU();

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

