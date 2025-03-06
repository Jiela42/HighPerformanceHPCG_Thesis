#ifndef STRIPED_COLORING_CUH
#define STRIPED_COLORING_CUH

#include "HPCGLib.hpp"
#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/striped_Matrix.hpp"
#include "UtilLib/cuda_utils.hpp"
#include "HPCG_versions/striped_warp_reduction.cuh"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

template <typename T>
class striped_coloring_Implementation : public striped_warp_reduction_Implementation<T> {
public:

    striped_coloring_Implementation(){

        // overwritting the inherited variables

        this->version_name = "Striped coloring (pre-computing COR Format)";
        // this->additional_parameters = "SymGS cooperation number = 16";

        this->implementation_type = Implementation_Type::STRIPED;

        this->SymGS_implemented = true;
        this->MG_implemented = true;
        this->CG_implemented = true;
        this->WAXPBY_implemented = false;
        this->Dot_implemented = false;
        this->SPMV_implemented = false;
        
    }

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device

    ) override {
        striped_coloring_computeSymGS(A, x_d, y_d);
    }

    

private:

    void striped_coloring_computeSymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method
__global__ void striped_coloring_half_SymGS_kernel(
    int color, int * color_pointer, int * color_sorted_rows,
    int num_rows, int num_cols,
    int num_stripes, int diag_offset,
    int * j_min_i,
    double * striped_A,
    double * x, double * y
);

#endif // STRIPED_COLORING_CUH