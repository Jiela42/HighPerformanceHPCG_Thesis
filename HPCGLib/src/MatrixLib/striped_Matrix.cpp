#include "MatrixLib/striped_Matrix.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "MatrixLib/coloring.cuh"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"

#include <vector>
#include <iostream>
#include <string>

#include <memory>

// we include this for comparison and being able to print the matrix types.
std::string string(MatrixType mt){
    switch (mt) {
        case MatrixType::Stencil_3D27P:
            return "Stencil_3D27P";
        default:
            return "UNKNOWN";
    }
}


// #include <stdio.h>

template <typename T>
striped_Matrix<T>::striped_Matrix() {
    this->nx = 0;
    this->ny = 0;
    this->nz = 0;
    this->nnz = 0;
    this->diag_index = -1;
    this->matrix_type = MatrixType::UNKNOWN;

    this->num_rows = 0;
    this->num_cols = 0;
    this->num_stripes = 0;


    this->j_min_i.clear();
    this->values.clear();
    this->j_min_i_d = nullptr;
    this->values_d = nullptr;

    this->color_pointer_d = nullptr;
    this->color_sorted_rows_d = nullptr;

    this->num_MG_pre_smooth_steps = 1;
    this->num_MG_post_smooth_steps = 1;
    this->coarse_Matrix = nullptr;
    this->f2c_op_d = nullptr;
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
    this->f2c_op.clear();

    this->CSR = nullptr;

}

template <typename T>
striped_Matrix<T>::~striped_Matrix(){

    if(this->j_min_i_d != nullptr){
        CHECK_CUDA(cudaFree(this->j_min_i_d));
        this->j_min_i_d = nullptr;
    }

    if(this->values_d != nullptr){
        CHECK_CUDA(cudaFree(this->values_d));
        this->values_d = nullptr;
    }

    if (this->color_pointer_d != nullptr) {
        CHECK_CUDA(cudaFree(this->color_pointer_d));
        this->color_pointer_d = nullptr;

    }
    if (this->color_sorted_rows_d != nullptr) {
        CHECK_CUDA(cudaFree(this->color_sorted_rows_d));
        this->color_sorted_rows_d = nullptr;
    }

    if (this->f2c_op_d != nullptr) {
        CHECK_CUDA(cudaFree(this->f2c_op_d));
        this->f2c_op_d = nullptr;
    }
    if (this->rc_d != nullptr) {
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
    
    if(this->CSR != nullptr){
        // this deletion causes a deadlock (because CSR points to Striped and vice versa)
        sparse_CSR_Matrix<T> *temp = this->CSR;
        this->CSR = nullptr;
        // we also have to set this matrix to null in our sibling matrix
        temp->Striped = nullptr;
        // when we delete the matrix, we also delete any coarse matrices, so we have to set that to null_ptr as well
        // we will delete the cooarse matrix of the corresponding CSR matrix, when we delete the coarse striped matrix

        // I don't like this. I'd rather cut the connection between this->coarse_Matrix and its CSR Matrix
        // however we do need to verify that these are all the same
        // which is done partially when we create the striped from the CSR,
        // however if coarse matrices are created later, the corresponding matrix, would not contain them
        temp->coarse_Matrix = nullptr;
        delete temp;
    }
    if(this->coarse_Matrix != nullptr){
        delete this->coarse_Matrix;
        this->coarse_Matrix = nullptr;
    }

}

template <typename T>
void striped_Matrix<T>::set_CSR(sparse_CSR_Matrix<T> *A){
    // std::cout << "set_CSR: setting CSR matrix" << std::endl;
    this->CSR = A;
}

template <typename T>
sparse_CSR_Matrix<T>* striped_Matrix<T>::get_CSR(){
    if(this->CSR == nullptr){
        // std::cout << "get_CSR: creating new CSR matrix" << std::endl;
        sparse_CSR_Matrix<T> *A = new sparse_CSR_Matrix<T>();
        A->sparse_CSR_Matrix_from_striped(*this);
    }
    return this->CSR;
}

template <typename T>
void striped_Matrix<T>::striped_Matrix_from_sparse_CSR(sparse_CSR_Matrix<T>& A){
    // std::cout << "entering striped_Matrix_from_sparse_CSR for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << std::endl;
    this->CSR = &A;
    A.set_Striped(this);
    if(A.get_matrix_type() == MatrixType::Stencil_3D27P and A.get_values_d() != nullptr and A.get_col_idx_d() != nullptr and A.get_row_ptr_d() != nullptr){
        // std::cout << "matrix is on GPU completely" << std::endl;
        this->striped_3D27P_Matrix_from_CSR_onGPU(A);
    }
    else if (A.get_matrix_type() == MatrixType::Stencil_3D27P) {
        this->striped_3D27P_Matrix_from_CSR_onCPU(A);
    } else {
        printf("ERROR: Unsupported matrix type for conversion to striped matrix\n");
        exit(1);
    }
}

template <typename T>
void striped_Matrix<T>::striped_3D27P_Matrix_from_CSR_onCPU(sparse_CSR_Matrix<T>& A){

    // std::cout << "striped_3D27P_Matrix_from_CSR (on CPU)" << std::endl;
    
    assert(A.get_matrix_type() == MatrixType::Stencil_3D27P);
    this->matrix_type = MatrixType::Stencil_3D27P;
    this->nx = A.get_nx();
    this->ny = A.get_ny();
    this->nz = A.get_nz();
    this->nnz = A.get_nnz();
    this->num_rows = A.get_num_rows();
    this->num_cols = A.get_num_cols();
    this->num_MG_pre_smooth_steps = A.get_num_MG_pre_smooth_steps();
    this->num_MG_post_smooth_steps = A.get_num_MG_post_smooth_steps();
    this->num_stripes = 27;
    this->j_min_i = std::vector<local_int_t>(this->num_stripes, 0);
    this->values = std::vector<T>(this->num_stripes * this->num_rows, 0);

    this->j_min_i_d = nullptr;
    this->values_d = nullptr;

    this->color_pointer_d = nullptr;
    this->color_sorted_rows_d = nullptr;

    this->coarse_Matrix = nullptr;
    this->f2c_op.clear();
    this->f2c_op_d = nullptr;
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;

    // first we make our mapping for the j_min_i
    // each point has num_stripe neighbours and each is associated with a coordinate relative to the point
    // the point itself is a neighobour, too {0,0,0}
    int neighbour_offsets [num_stripes][3] = {
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

        int off_x = neighbour_offsets[i][0];
        int off_y = neighbour_offsets[i][1];
        int off_z = neighbour_offsets[i][2];
        
        this->j_min_i[i] = off_x + off_y * this->nx + off_z * this->nx * this->ny;
        if (this->j_min_i[i] == 0) {
            this->diag_index = i;
        }
    }

    local_int_t elem_ctr = 0;

    // now that we have the static offsets which define i & j, we can make the actual matrix
    for (local_int_t i = 0; i < this->num_rows; i++) {
        for (int stripe_j = 0; stripe_j < this->num_stripes; stripe_j++) {
            local_int_t j = this->j_min_i[stripe_j] + i;
            // check if j is in bounds (since not every point has all 27 neighbours)
            if (j >= 0 && j < this->num_cols) {
                double elem = A.get_element(i, j);
                // also make sure we don't add zero elements
                if(elem != 0){
                    this->values[i * this->num_stripes + stripe_j] = elem;
                    elem_ctr++;
                }
            }
        }
    }
    assert(elem_ctr == this->nnz);
    if(A.get_coarse_Matrix() != nullptr){
        this->coarse_Matrix = new striped_Matrix<T>();
        this->coarse_Matrix->striped_Matrix_from_sparse_CSR(*(A.get_coarse_Matrix()));
    }
    // we also gotta copy the f2c operator
    if (A.get_f2c_op().size() > 0) {
        this->f2c_op = A.get_f2c_op();
    } else {
        this->f2c_op.clear();
    }    
}

template <typename T>
void striped_Matrix<T>::Generate_striped_3D27P_Matrix_onGPU(int nx, int ny, int nz) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;

    this->matrix_type = MatrixType::Stencil_3D27P;

    // a single gpu setting
    Problem *p = new Problem;
    GenerateProblem(1, 1, 1, this->nx, this->ny, this->nz, 1, 0, p);

    this->num_rows = p->nx * p->ny * p->nz; 
    this->num_cols = p->nx * p->ny * p->nz;
    this->num_stripes = 27;

    local_int_t num_interior_points = (nx - 2) * (ny - 2) * (nz - 2);
    local_int_t num_face_points = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2));
    local_int_t num_edge_points = 4 * ((nx - 2) + (ny - 2) + (nz - 2));
    local_int_t num_corner_points = 8;

    local_int_t nnz_interior = 27 * num_interior_points;
    local_int_t nnz_face = 18 * num_face_points;
    local_int_t nnz_edge = 12 * num_edge_points;
    local_int_t nnz_corner = 8 * num_corner_points;

    local_int_t nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner;

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
    this->rc_d = nullptr;
    this->xc_d = nullptr;
    this->Axf_d = nullptr;
    //this->f2c_op.clear();

    CHECK_CUDA(cudaMalloc(&this->values_d, sizeof(double) * num_rows* 27));
    GenerateStripedPartialMatrix_GPU(p, this->values_d);

    // fill j_min_i_d
    int neighbour_offsets [num_stripes][3] = {
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
        int off_x = neighbour_offsets[i][0];
        int off_y = neighbour_offsets[i][1];
        int off_z = neighbour_offsets[i][2];
        
        this->j_min_i[i] = off_x + off_y * p->gnx + off_z * p->gnx * p->gny;
        if (this->j_min_i[i] == 0) {
            this->diag_index = i;
        }
    }

    // if(this->j_min_i.size() != this->num_stripes){
    //     std::cout << "j_min_i size: " << this->j_min_i.size() << std::endl;
    //     std::cout << "num_stripes: " << this->num_stripes << std::endl;
    // }

    // for(int i = 0; i < this->num_stripes; i++){
    //     std::cout << "j_min_i[" << i << "]: " << this->j_min_i[i] << std::endl;

    //     // now also print the stuff .data points to
    //     std::cout << "j_min_i.data[" << i << "]: " << this->j_min_i.data()[i] << std::endl;
    // }

    std::vector<local_int_t> j_min_i_h(this->num_stripes, 0);

    for(int i = 0; i < this->num_stripes; i++){
        j_min_i_h[i] = this->j_min_i[i];
    }


    // std::cout << "Alignment of j_min_i.data(): "
    //         << reinterpret_cast<uintptr_t>(this->j_min_i.data()) % alignof(int)
    //         << std::endl;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    assert(this->j_min_i.size() == this->num_stripes);
    // this->j_min_i.resize(this->num_stripes);
    CHECK_CUDA(cudaMalloc(&this->j_min_i_d, this->num_stripes * sizeof(local_int_t)));
    // std::cout << "Device pointer j_min_i_d: " << this->j_min_i_d << std::endl;

    // std::cout << "Allocated j_min_i_d with size: " << this->num_stripes * sizeof(int) << " bytes" << std::endl;
    CHECK_CUDA(cudaMemcpy(this->j_min_i_d, this->j_min_i.data(), this->num_stripes * sizeof(local_int_t), cudaMemcpyHostToDevice));

    // we don't need the host-side j_min_i anymore
    this->j_min_i.clear();

}

template <typename T>
void striped_Matrix<T>::striped_3D27P_Matrix_from_CSR_onGPU(sparse_CSR_Matrix<T>& A){
    
    // std::cout << "striped_3D27P_Matrix_from_CSR_onGPU" << std::endl;

    assert(A.get_matrix_type() == MatrixType::Stencil_3D27P);

    // first we make sure that the matrix is on the GPU
    if(A.get_col_idx_d() == nullptr or A.get_row_ptr_d() == nullptr or A.get_values_d() == nullptr){
        // we print a warning, because this should really not happen
        printf("WARNING: sparse_CSR_Matrix not on GPU or only partially on GPU\n");
        printf("We will generate the matrix on the GPU\n");
        printf("This frees the pointer to the matrix, should any of them be non-nullptr\n");
        // should there be a bug and one of the pointers to the GPU isn't a nullpointer, we need to free it
        A.remove_Matrix_from_GPU();
        // now we generate the matrix on the GPU
        A.generateMatrix_onGPU(A.get_nx(), A.get_ny(), A.get_nz());
    }

    this->matrix_type = MatrixType::Stencil_3D27P;
    this->nx = A.get_nx();
    this->ny = A.get_ny();
    this->nz = A.get_nz();
    this->nnz = A.get_nnz();
    this->num_rows = A.get_num_rows();
    this->num_cols = A.get_num_cols();
    this->num_MG_pre_smooth_steps = A.get_num_MG_pre_smooth_steps();
    this->num_MG_post_smooth_steps = A.get_num_MG_post_smooth_steps();
    this->num_stripes = 27;
    this->j_min_i = std::vector<local_int_t>(this->num_stripes, 0);
    this->values = std::vector<T>(this->num_stripes * this->num_rows, 0);
    this->j_min_i_d = nullptr;
    this->values_d = nullptr;
    this->color_pointer_d = nullptr;
    this->color_sorted_rows_d = nullptr;
    this->coarse_Matrix = nullptr;
    this->f2c_op.clear();

    // first we make our mapping for the j_min_i (on the CPU)
    // each point has num_stripe neighbours and each is associated with a coordinate relative to the point
    // the point itself is a neighobour, too {0,0,0}
    int neighbour_offsets [num_stripes][3] = {
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

        int off_x = neighbour_offsets[i][0];
        int off_y = neighbour_offsets[i][1];
        int off_z = neighbour_offsets[i][2];
        
        this->j_min_i[i] = off_x + off_y * this->nx + off_z * this->nx * this->ny;
        if (this->j_min_i[i] == 0) {
            this->diag_index = i;
        }
    }

    // now we allocate the space on the GPU
    CHECK_CUDA(cudaMalloc(&this->j_min_i_d, this->num_stripes * sizeof(local_int_t)));
    CHECK_CUDA(cudaMalloc(&this->values_d, this->num_stripes * this->num_rows * sizeof(T)));

    // we copy the j_min_i onto the GPU
    CHECK_CUDA(cudaMemcpy(this->j_min_i_d, this->j_min_i.data(), this->num_stripes * sizeof(local_int_t), cudaMemcpyHostToDevice));

    // set the values on the GPU to zero
    CHECK_CUDA(cudaMemset(this->values_d, 0, this->num_stripes * this->num_rows * sizeof(T)));

    // call a function to generate the values on the GPU
    local_int_t counted_nnz = generate_striped_3D27P_Matrix_from_CSR(
        this->nx, this->ny, this->nz,
        A.get_row_ptr_d(), A.get_col_idx_d(), A.get_values_d(),
        this->num_stripes, this->j_min_i_d, this->values_d);
    
    // std::cout << "nx: " << this->nx << std::endl;
    // std::cout << "ny: " << this->ny << std::endl;
    // std::cout << "nz: " << this->nz << std::endl;
    // std::cout << "counted_nnz: " << counted_nnz << std::endl;
    // std::cout << "nnz: " << this->nnz << std::endl;

    assert(counted_nnz == this->nnz);

    // since this is the GPU generation the host side buffers need to be empty
    this->values.clear();
    this->j_min_i.clear();


    // if the CSR matrix has a coarse matrix, we also need to generate the coarse matrix for the striped matrix
    if(A.get_coarse_Matrix() != nullptr){
        this->coarse_Matrix = new striped_Matrix<T>();
        this->coarse_Matrix->striped_Matrix_from_sparse_CSR(*(A.get_coarse_Matrix()));
    }

    if(A.get_f2c_op_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->f2c_op_d, this->num_rows * sizeof(local_int_t)));
        CHECK_CUDA(cudaMemcpy(this->f2c_op_d, A.get_f2c_op_d(), this->num_rows * sizeof(local_int_t), cudaMemcpyDeviceToDevice));
    } else{
        this->f2c_op_d = nullptr;
    }
    if(A.get_rc_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->rc_d, this->num_rows * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(this->rc_d, A.get_rc_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToDevice));
    } else{
        this->rc_d = nullptr;
    }
    if(A.get_xc_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->xc_d, this->num_rows * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(this->xc_d, A.get_xc_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToDevice));
    } else{
        this->xc_d = nullptr;
    }

    local_int_t num_fine_rows = this->nx * 2 * this->ny * 2 * this->nz * 2;
    if(A.get_Axf_d() != nullptr){
        CHECK_CUDA(cudaMalloc(&this->Axf_d, num_fine_rows * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(this->Axf_d, A.get_Axf_d(), num_fine_rows * sizeof(T), cudaMemcpyDeviceToDevice));
    } else{
        this->Axf_d = nullptr;
    }

}

template <typename T>
void striped_Matrix<T>::copy_Matrix_toGPU(){

    //we delete the old data from the GPU
    this->remove_Matrix_from_GPU();
    
    // we copy the j_min_i to the GPU
    CHECK_CUDA(cudaMalloc(&this->j_min_i_d, this->num_stripes * sizeof(local_int_t)));
    CHECK_CUDA(cudaMemcpy(this->j_min_i_d, this->j_min_i.data(), this->num_stripes * sizeof(local_int_t), cudaMemcpyHostToDevice));

    // we copy the values to the GPU
    CHECK_CUDA(cudaMalloc(&this->values_d, this->num_stripes * this->num_rows * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(this->values_d, this->values.data(), this->num_stripes * this->num_rows * sizeof(T), cudaMemcpyHostToDevice));

    if(this->coarse_Matrix != nullptr){
        this->coarse_Matrix->copy_Matrix_toGPU();
    }

    if(this->f2c_op_d != nullptr){
        CHECK_CUDA(cudaMalloc(&this->f2c_op_d, this->num_rows * sizeof(local_int_t)));
        CHECK_CUDA(cudaMemcpy(this->f2c_op_d, this->f2c_op.data(), this->num_rows * sizeof(local_int_t), cudaMemcpyHostToDevice));
    }

}

template <typename T>
void striped_Matrix<T>::copy_Matrix_toCPU(){

    // rezise the vectors
    this->j_min_i.resize(this->num_stripes);
    this->values.resize(this->num_stripes * this->num_rows);

    // std::cout << "num_stripes: " << this->num_stripes << std::endl;
    // std::cout << "num_rows: " << this->num_rows << std::endl;

    // we copy the j_min_i to the CPU
    CHECK_CUDA(cudaMemcpy(this->j_min_i.data(), this->j_min_i_d, this->num_stripes * sizeof(local_int_t), cudaMemcpyDeviceToHost));

    // we copy the values to the CPU
    CHECK_CUDA(cudaMemcpy(this->values.data(), this->values_d, this->num_stripes * this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));

    if(this->f2c_op_d != nullptr){
        this->f2c_op.resize(this->num_rows);
        CHECK_CUDA(cudaMemcpy(this->f2c_op.data(), this->f2c_op_d, this->num_rows * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    }

    if(this->coarse_Matrix != nullptr){
        this->coarse_Matrix->copy_Matrix_toCPU();
    }

}

template <typename T>
void striped_Matrix<T>::remove_Matrix_from_GPU(){
    if(this->j_min_i_d != nullptr){
        CHECK_CUDA(cudaFree(this->j_min_i_d));
        this->j_min_i_d = nullptr;
    }

    if(this->values_d != nullptr){
        CHECK_CUDA(cudaFree(this->values_d));
        this->values_d = nullptr;
    }
    if(this->color_pointer_d != nullptr){
        CHECK_CUDA(cudaFree(this->color_pointer_d));
        this->color_pointer_d = nullptr;
    }
    if(this->color_sorted_rows_d != nullptr){
        CHECK_CUDA(cudaFree(this->color_sorted_rows_d));
        this->color_sorted_rows_d = nullptr;
    }
    if(this->f2c_op_d != nullptr){
        CHECK_CUDA(cudaFree(this->f2c_op_d));
        this->f2c_op_d = nullptr;
    }
}

template <typename T>
striped_Matrix<T>* striped_Matrix<T>::get_coarse_Matrix(){
    return this->coarse_Matrix;
}

template <typename T>
T* striped_Matrix<T>::get_rc_d(){
    return this->rc_d;
}

template <typename T>
T* striped_Matrix<T>::get_xc_d(){
    return this->xc_d;
}


template <typename T>
T* striped_Matrix<T>::get_Axf_d(){
    return this->Axf_d;
}


template <typename T>
void striped_Matrix<T>::generate_coloring(){

    if(this->matrix_type == MatrixType::Stencil_3D27P){
        local_int_t num_colors = (nx -1) + 2 * (ny-1) + 4 * (nz-1) + 1;
    
        // first we allocate the space on the GPU
        CHECK_CUDA(cudaMalloc(&this->color_pointer_d, (num_colors + 1) * sizeof(local_int_t)));
        CHECK_CUDA(cudaMalloc(&this->color_sorted_rows_d, this->num_rows * sizeof(local_int_t)));
    
        get_color_row_mapping(this->nx, this->ny, this->nz, this->color_pointer_d, this->color_sorted_rows_d);
    
        // also generate the coloring for any coarse matrices
        if(this->coarse_Matrix != nullptr){
            this->coarse_Matrix->generate_coloring();
        }
    } else{
        printf("ERROR: Unsupported matrix type for coloring\n");
        exit(1);
    }

}

template <typename T>
void striped_Matrix<T>::generate_box_coloring(){

    if(this->matrix_type == MatrixType::Stencil_3D27P){

        int num_colors = 27;

        // first we allocate the space on the GPU
        // to do so we first check if the color ptr are nullptr
        if(this->color_pointer_d != nullptr){
            // if they are not nullptr, we print a warning and free them
            std::cerr << "WARNING: color_pointer_d not nullptr, freeing it to generate box coloring" << std::endl;
            CHECK_CUDA(cudaFree(this->color_pointer_d));
        }
        if(this->color_sorted_rows_d != nullptr){
            // if they are not nullptr, we print a warning and free them
            std::cerr << "WARNING: color_sorted_rows_d not nullptr, freeing it to generate box coloring" << std::endl;
            CHECK_CUDA(cudaFree(this->color_sorted_rows_d));
        }

        // now we can safely allocate the space for the pointers
        CHECK_CUDA(cudaMalloc(&this->color_pointer_d, (num_colors + 1) * sizeof(local_int_t)));
        CHECK_CUDA(cudaMalloc(&this->color_sorted_rows_d, this->num_rows * sizeof(local_int_t)));

        get_color_row_mapping_for_boxColoring(this->nx, this->ny, this->nz, this->color_pointer_d, this->color_sorted_rows_d);

        // also generate the coloring for any coarse matrices
        if(this->coarse_Matrix != nullptr){
            this->coarse_Matrix->generate_box_coloring();
        }
        
    } else{
        printf("ERROR: Unsupported matrix type for coloring\n");
        exit(1);
    }

}

template <typename T>
local_int_t* striped_Matrix<T>::get_color_pointer_d(){
    return this->color_pointer_d;
}

template <typename T>
local_int_t* striped_Matrix<T>::get_color_sorted_rows_d(){
    return this->color_sorted_rows_d;
}

template <typename T>
std::vector<local_int_t> striped_Matrix<T>::get_color_pointer_vector(){

    local_int_t num_colors = (this->nx-1) + 2 * (this->ny -1) + 4 *(this->nz-1) + 1;
    std::vector<local_int_t> color_pointer(num_colors + 1, 0);
    CHECK_CUDA(cudaMemcpy(color_pointer.data(), this->color_pointer_d, (num_colors + 1) * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    return color_pointer;
}

template <typename T>
std::vector<local_int_t> striped_Matrix<T>::get_color_sorted_rows_vector(){

    std::vector<local_int_t> color_sorted_rows(this->num_rows, 0);
    CHECK_CUDA(cudaMemcpy(color_sorted_rows.data(), this->color_sorted_rows_d, this->num_rows * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    return color_sorted_rows;
}

template <typename T>
void striped_Matrix<T>::print_COR_Format(){
    local_int_t max_color = (this->nx-1) + 2 * (this->ny-1) + 4 * (this->nz -1) + 1;
    ::print_COR_Format(max_color, this->num_rows, this->color_pointer_d, this->color_sorted_rows_d);
}

template <typename T>
local_int_t striped_Matrix<T>::get_num_rows() const{
    return this->num_rows;
}

template <typename T>
local_int_t striped_Matrix<T>::get_num_cols() const{
    return this->num_cols;
}

template <typename T>
int striped_Matrix<T>::get_ny() const{
    return this->ny;
}

template <typename T>
int striped_partial_Matrix<T>::get_nx() const{
    return this->nx;
}

template <typename T>
int striped_partial_Matrix<T>::get_ny() const{
    return this->ny;
}

template <typename T>
int striped_partial_Matrix<T>::get_nz() const{
    return this->nz;
}


template <typename T>
local_int_t striped_Matrix<T>::get_nnz() const{
    return this->nnz;
}

template <typename T>
int striped_Matrix<T>::get_num_MG_pre_smooth_steps() const{
    return this->num_MG_pre_smooth_steps;
}

template <typename T>
int striped_Matrix<T>::get_num_MG_post_smooth_steps() const{
    return this->num_MG_post_smooth_steps;
}





template <typename T>
MatrixType striped_Matrix<T>::get_matrix_type() const{
    return this->matrix_type;
}

template <typename T>
std::vector<local_int_t>& striped_Matrix<T>::get_j_min_i(){
    return this->j_min_i;
}

template <typename T>
std::vector<T>& striped_Matrix<T>::get_values(){
    return this->values;
}

template <typename T>
local_int_t * striped_Matrix<T>::get_j_min_i_d(){
    return this->j_min_i_d;
}

template <typename T>
T * striped_Matrix<T>::get_values_d(){
    return this->values_d;
}

template <typename T>
std::vector<local_int_t> striped_Matrix<T>::get_f2c_op(){
    return this->f2c_op;
}

template <typename T>
local_int_t * striped_Matrix<T>::get_f2c_op_d(){
    return this->f2c_op_d;
}

template <typename T>
void striped_Matrix<T>::generate_f2c_operator_onGPU() {
    const int x = this->nx;
    const int y = this->ny;
    const int z = this->nz;
    const local_int_t fine_n_rows = x *2 * y * 2 * z * 2;

    // allocate space for the device pointers
    CHECK_CUDA(cudaMalloc(&this->f2c_op_d, fine_n_rows * sizeof(local_int_t)));

    // set them to zero
    CHECK_CUDA(cudaMemset(this->f2c_op_d, 0, fine_n_rows * sizeof(local_int_t)));

    generate_f2c_operator(x, y, z, x*2, y*2, z*2, f2c_op_d);

    // generate_partialf2c_operator(x, y, z, x*2, y*2, z*2, f2c_op_d);
}

template <typename T>
void striped_Matrix<T>::init_coarse_buffer(){
    CHECK_CUDA(cudaMalloc(&this->rc_d, this->num_rows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&this->xc_d, this->num_rows * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&this->Axf_d, this->num_rows*8 * sizeof(T)));

    CHECK_CUDA(cudaMemset(this->rc_d, 0, this->num_rows * sizeof(T)));
    CHECK_CUDA(cudaMemset(this->xc_d, 0, this->num_rows * sizeof(T)));
    CHECK_CUDA(cudaMemset(this->Axf_d, 0, this->num_rows*8 * sizeof(T)));
}

template <typename T>
void striped_Matrix<T>::initialize_coarse_Matrix(){
    assert(this->nx % 2 == 0);
    assert(this->ny % 2 == 0);
    assert(this->nz % 2 == 0);
    assert(this->coarse_Matrix == nullptr);

    int nx_f = this->nx;
    int ny_f = this->ny;
    int nz_f = this->nz;
    //int fine_n_rows = this->nx *2 * this->ny * 2 * this->nz * 2;
    int nx_c = this->nx / 2;
    int ny_c = this->ny / 2;
    int nz_c = this->nz / 2;

    // allocate coarse matrix
    this->coarse_Matrix= new striped_Matrix<T>();
    this->coarse_Matrix->Generate_striped_3D27P_Matrix_onGPU(nx_c, ny_c, nz_c);
    this->coarse_Matrix->generate_f2c_operator_onGPU();

    // allocate halos rc, xc, Axf and set to zero
    init_coarse_buffer();
}

template <typename T>
T striped_Matrix<T>::get_element(local_int_t i, local_int_t j) const{

    // check if j is in bounds (since not every point has all 27 neighbours)
    if (j >= 0 && j < this->num_cols) {
        for (int stripe_j = 0; stripe_j < this->num_stripes; stripe_j++) {
            if (j == i + this->j_min_i[stripe_j]) {
                return this->values[i * this->num_stripes + stripe_j];
            }
        }
    }
    // printf("WARNING Element row %d, col %d not found\n", i, j);
    return T();
}

template <typename T>
int striped_Matrix<T>::get_diag_index() const{
    return this->diag_index;
}

template <typename T>
void striped_Matrix<T>::set_num_rows(local_int_t num_rows){
    // std::cout << "random debug prints here:" << std::endl;
    // std::cout << "this->csr: " << this->CSR << std::endl;
    this->num_rows = num_rows;
}

template<typename T>
bool striped_Matrix<T>::compare_to(striped_Matrix<T>& other){

    std::cout << "comparing striped matrices of size " << this->nx << "x"<< this->ny << "x" << this->nz << std::endl;
    return true;

    // // check all the single values
    // if(this->nx != other.get_nx()){
    //     std::cout << "nx not equal" << std::endl;
    //     std::cout << "this->nx: " << this->nx << std::endl;
    //     std::cout << "other.get_nx(): " << other.get_nx() << std::endl;
    //     return false;
    // }
    // if(this->ny != other.get_ny()){
    //     std::cout << "ny not equal" << std::endl;
    //     std::cout << "this->ny: " << this->ny << std::endl;
    //     std::cout << "other.get_ny(): " << other.get_ny() << std::endl;
    //     return false;
    // }
    // if(this->nnz != other.get_nnz()){
    //     std::cout << "nnz not equal" << std::endl;
    //     std::cout << "this->nnz: " << this->nnz << std::endl;
    //     std::cout << "other.get_nnz(): " << other.get_nnz() << std::endl;
    //     return false;
    // }
    // if(this->diag_index != other.get_diag_index()){
    //     std::cout << "diag_index not equal" << std::endl;
    //     std::cout << "this->diag_index: " << this->diag_index << std::endl;
    //     std::cout << "other.get_diag_index(): " << other.get_diag_index() << std::endl;
    //     return false;
    // }
    // if(this->matrix_type != other.get_matrix_type()){
    //     std::cout << "matrix_type not equal" << std::endl;
    //     std::cout << "this->matrix_type: " << string(this->matrix_type) << std::endl;
    //     std::cout << "other.get_matrix_type(): " << string(other.get_matrix_type()) << std::endl;
    //     return false;
    // }
    // if(this->num_MG_pre_smooth_steps != other.get_num_MG_pre_smooth_steps()){
    //     std::cout << "num_MG_pre_smooth_steps not equal" << std::endl;
    //     std::cout << "this->num_MG_pre_smooth_steps: " << this->num_MG_pre_smooth_steps << std::endl;
    //     std::cout << "other.get_num_MG_pre_smooth_steps(): " << other.get_num_MG_pre_smooth_steps() << std::endl;
    //     return false;
    // }
    // if(this->num_MG_post_smooth_steps != other.get_num_MG_post_smooth_steps()){
    //     std::cout << "num_MG_post_smooth_steps not equal" << std::endl;
    //     std::cout << "this->num_MG_post_smooth_steps: " << this->num_MG_post_smooth_steps << std::endl;
    //     std::cout << "other.get_num_MG_post_smooth_steps(): " << other.get_num_MG_post_smooth_steps() << std::endl;
    //     return false;
    // }


    // if(this->num_rows != other.get_num_rows()){
    //     std::cout << "num_rows not equal" << std::endl;
    //     return false;
    // }
    // if(this->num_cols != other.get_num_cols()){
    //     std::cout << "num_cols not equal" << std::endl;
    //     return false;
    // }
    // if(this->num_stripes != other.get_num_stripes()){
    //     std::cout << "num_stripes not equal" << std::endl;
    //     return false;
    // }
    // if(this->j_min_i != other.get_j_min_i()){
    //     std::cout << "j_min_i not equal" << std::endl;
    //     std::cout << "this->j_min_i: ";
    //     for(auto i : this->j_min_i){
    //         std::cout << i << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "other.get_j_min_i(): ";
    //     for(auto i : other.get_j_min_i()){
    //         std::cout << i << " ";
    //     }
    //     std::cout << std::endl;
    //     return false;
    // }
    // if(this->values != other.get_values()){
    //     std::cout << "values not equal" << std::endl;
    //     std::cout << "this->values: ";
    //     for(auto i : this->values){
    //         std::cout << i << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "other.get_values(): ";
    //     for(auto i : other.get_values()){
    //         std::cout << i << " ";
    //     }
    //     std::cout << std::endl;

    //     return false;
    // }

    // if(this->f2c_op != other.get_f2c_op()){
    //     std::cout << "f2c_op not equal" << std::endl;
    //     return false;
    // }

    // // also check if the device stuff is equal (if it exists)

    // if(this->j_min_i_d != nullptr && other.get_j_min_i_d() != nullptr){
    //     // at least one of them is not null
    //     if (!this->j_min_i_d || !other.get_j_min_i_d()){
    //         // one of them is null -> one is null the other isn't
    //         std::cout << "j_min_i_d not equal" << std::endl;
    //         return false;
    //     } else {
    //         std::vector<local_int_t> A_j_min_i(this->num_stripes, 0);
    //         std::vector<local_int_t> B_j_min_i(this->num_stripes, 0);
    //         CHECK_CUDA(cudaMemcpy(A_j_min_i.data(), this->j_min_i_d, this->num_stripes * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    //         CHECK_CUDA(cudaMemcpy(B_j_min_i.data(), other.get_j_min_i_d(), this->num_stripes * sizeof(local_int_t), cudaMemcpyDeviceToHost));
        
    //         if(A_j_min_i != B_j_min_i){
    //             std::cout << "j_min_i_d not equal" << std::endl;
    //             return false;
    //         }
    //     }
    // }

    // if(this->values_d != nullptr && other.get_values_d() != nullptr){
    //     // at least one of them is not null
    //     if (!this->values_d || !other.get_values_d()){
    //         // one of them is null -> one is null the other isn't
    //         std::cout << "values_d not equal" << std::endl;
    //         return false;
    //     } else {
    //         std::vector<T> A_values(this->num_stripes * this->num_rows, 0);
    //         std::vector<T> B_values(this->num_stripes * this->num_rows, 0);
    //         CHECK_CUDA(cudaMemcpy(A_values.data(), this->values_d, this->num_stripes * this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
    //         CHECK_CUDA(cudaMemcpy(B_values.data(), other.get_values_d(), this->num_stripes * this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
        
    //         if(A_values != B_values){
    //             std::cout << "values_d not equal" << std::endl;
    //             return false;
    //         }
    //     }
    // }

    // if(this->f2c_op_d != nullptr && other.get_f2c_op_d() != nullptr){
    //     // at least one of them is not null
    //     if (!this->f2c_op_d || !other.get_f2c_op_d()){
    //         // one of them is null -> one is null the other isn't
    //         std::cout << "f2c_op_d not equal" << std::endl;
    //         return false;
    //     } else {

    //         std::vector<local_int_t> A_f2c_op(this->num_rows, 0);
    //         std::vector<local_int_t> B_f2c_op(this->num_rows, 0);
    //         CHECK_CUDA(cudaMemcpy(A_f2c_op.data(), this->f2c_op_d, this->num_rows * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    //         CHECK_CUDA(cudaMemcpy(B_f2c_op.data(), other.get_f2c_op_d(), this->num_rows * sizeof(local_int_t), cudaMemcpyDeviceToHost));
        
    //         if(A_f2c_op != B_f2c_op){
    //             std::cout << "f2c_op_d not equal" << std::endl;
    //             return false;
    //         }
    //     }
    // }

    // // check if the rc_d is equal
    // if(this->rc_d != nullptr && other.get_rc_d() != nullptr){
    //     // at least one of them is not null
    //     if (!this->rc_d || !other.get_rc_d()){
    //         // one of them is null -> one is null the other isn't
    //         std::cout << "rc_d not equal" << std::endl;
    //         return false;
    //     }
    //     else{
    //         std::vector<T> A_rc(this->num_rows, 0);
    //         std::vector<T> B_rc(this->num_rows, 0);
    //         CHECK_CUDA(cudaMemcpy(A_rc.data(), this->rc_d, this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
    //         CHECK_CUDA(cudaMemcpy(B_rc.data(), other.get_rc_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
        
    //         if(A_rc != B_rc){
    //             std::cout << "rc_d not equal" << std::endl;
    //             return false;
    //         }
    //     }
    // }

    // // check if the xc_d is equal
    // if(this->xc_d != nullptr && other.get_xc_d() != nullptr){
    //     // at least one of them is not null
    //     if (!this->xc_d || !other.get_xc_d()){
    //         // one of them is null -> one is null the other isn't
    //         std::cout << "xc_d not equal" << std::endl;
    //         return false;
    //     }
    //     else{
    //         std::vector<T> A_xc(this->num_rows, 0);
    //         std::vector<T> B_xc(this->num_rows, 0);
    //         CHECK_CUDA(cudaMemcpy(A_xc.data(), this->xc_d, this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
    //         CHECK_CUDA(cudaMemcpy(B_xc.data(), other.get_xc_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
        
    //         if(A_xc != B_xc){
    //             std::cout << "xc_d not equal" << std::endl;
    //             return false;
    //         }
    //     }
    // }
    // // check if the Axf_d is equal
    // if(this->Axf_d != nullptr && other.get_Axf_d() != nullptr){
    //     // at least one of them is not null
    //     if (!this->Axf_d || !other.get_Axf_d()){
    //         // one of them is null -> one is null the other isn't
    //         std::cout << "Axf_d not equal" << std::endl;
    //         return false;
    //     }
    //     else{
    //         std::vector<T> A_Axf(this->num_rows, 0);
    //         std::vector<T> B_Axf(this->num_rows, 0);
    //         CHECK_CUDA(cudaMemcpy(A_Axf.data(), this->Axf_d, this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
    //         CHECK_CUDA(cudaMemcpy(B_Axf.data(), other.get_Axf_d(), this->num_rows * sizeof(T), cudaMemcpyDeviceToHost));
        
    //         if(A_Axf != B_Axf){
    //             std::cout << "Axf_d not equal" << std::endl;
    //             return false;
    //         }
    //     }
    // }



    // // check if the coarse matrix is equal
    // if(this->coarse_Matrix != nullptr && other.get_coarse_Matrix() != nullptr){
    //     // at least one of them is not null
    //     if(!this->coarse_Matrix->compare_to(*(other.get_coarse_Matrix()))){
    //         std::cout << "coarse matrix not equal" << std::endl;
    //         return false;
    //     }
    // } else if(this->coarse_Matrix != nullptr && other.get_coarse_Matrix() == nullptr){
    //     // one of them is null -> one is null the other isn't
    //     std::cout << "one of the coarse matrices is null, the other isn't mmrayu" << std::endl;

        
    //     std::cout << "this->coarse_Matrix: " << this->coarse_Matrix << std::endl;
    //     std::cout << "other.get_coarse_Matrix(): " << other.get_coarse_Matrix() << std::endl;
        

    //     return false;
    // } else if(this->coarse_Matrix == nullptr && other.get_coarse_Matrix() != nullptr){
    //     std::cout << "one of the coarse matrices is null, the other isn't" << std::endl;
    //     return false;
    // }

    return true;
}

template <typename T>
void striped_Matrix<T>::print() const{
    std::cout << "striped Matrix: " << std::endl;
    std::cout << "nx: " << this->nx << " ny: " << this->ny << " nz: " << this->nz << std::endl;
    std::cout << "num_rows: " << this->num_rows << " num_cols: " << this->num_cols << std::endl;
    std::cout << "num_stripes: " << this->num_stripes << std::endl;
    std::cout << "j_min_i: ";
    for (int i = 0; i < this->num_stripes; i++) {
        std::cout << this->j_min_i[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Values: ";
    for (local_int_t i = 0; i < this->num_rows; i++) {
        for (local_int_t j = 0; j < this->num_stripes; j++) {
            std::cout << this->values[i * this->num_stripes + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout<< "printing not implemented for coarse Matrix, if you want that, implement it" << std::endl;
}

// explicit template instantiation
template class striped_Matrix<double>;