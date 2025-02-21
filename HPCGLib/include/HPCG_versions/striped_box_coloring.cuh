#ifndef STRIPED_BOX_COLORING_CUH
#define STRIPED_BOX_COLORING_CUH

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
class striped_box_coloring_Implementation : public HPCG_functions<T> {
public:

    int bx, by, bz; // the box size in x, y and z direction
    int SymGS_cooperation_number; // the cooperation number for the SymGS

    striped_box_coloring_Implementation(){

        // overwritting the inherited variables

        this->version_name = "Striped Box coloring";
        this->norm_based = true;
        // this->additional_parameters = "SymGS cooperation number = 16";

        this->implementation_type = Implementation_Type::STRIPED;

        this->SymGS_implemented = true;

        // default box size for coloring
        this->bx = 3;
        this->by = 3;
        this->bz = 3;

        // set the default cooperation number for the SymGS
        this->SymGS_cooperation_number = 4;
        
    }

    void set_box_size(int bx, int by, int bz){
        this->bx = bx;
        this->by = by;
        this->bz = bz;
    }

    void compute_CG(
        striped_Matrix<T> & A,
        T * b_d, T * x_d,
        int & n_iters, T& normr, T& normr0) override {
        std::cerr << "Warning: compute_CG is not implemented in striped_box_coloring." << std::endl;
    }
    
    void compute_MG(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_MG is not implemented in striped_box_coloring." << std::endl;
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in striped_box_coloring." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_box_coloring_computeSymGS(A, x_d, y_d);
    }

    void compute_SPMV(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input in the striped_box_coloring Implementation.");
    }

    void compute_WAXPBY(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in striped_box_coloring." << std::endl;
    }
    void compute_WAXPBY(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "Warning: compute_WAXPBY is not implemented in striped_box_coloring." << std::endl;
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the sparse_CSR Matrix in striped_box_coloring." << std::endl;
    }

    void compute_Dot(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "Warning: compute_Dot is not implemented using the striped Matrix in striped_box_coloring." << std::endl;
    }


    // Striped matrices need a special SPMV implementations because they have special arguments
    // we have some aliasing going on depending on the input parameters.
    void compute_SPMV(
        striped_Matrix<T>& A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SPMV requires different arguments in striped_box_coloring." << std::endl;
    }

private:

    void striped_box_coloring_computeSymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method


#endif // STRIPED_BOX_COLORING_CUH