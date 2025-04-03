#ifndef STRIPED_BOX_COLORING_CUH
#define STRIPED_BOX_COLORING_CUH

#include "HPCGLib_hipified.hpp"
#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"
#include "MatrixLib/striped_Matrix_hipified.hpp"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "HPCG_versions/striped_warp_reduction_hipified.cuh"


#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <hip/hip_runtime.h>

template <typename T>
class striped_box_coloring_Implementation : public striped_warp_reduction_Implementation<T> {
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
        this->MG_implemented = true;
        this->CG_implemented = true;
        this->WAXPBY_implemented = false;
        this->Dot_implemented = false;
        this->SPMV_implemented = false;


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

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_box_coloring_computeSymGS(A, x_d, y_d);
    }

private:

    void striped_box_coloring_computeSymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method


#endif // STRIPED_BOX_COLORING_CUH