#ifndef STRIPED_COLORINGPRECOMPUTED_CUH
#define STRIPED_COLORINGPRECOMPUTED_CUH

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
class striped_coloringPrecomputed_Implementation : public striped_warp_reduction_Implementation<T> {
public:

    striped_coloringPrecomputed_Implementation(){

        // overwritting the inherited variables

        this->version_name = "Striped coloring (COR Format already stored on the GPU)";
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
        striped_coloringPrecomputed_computeSymGS(A, x_d, y_d);
    }


private:

    // we use the CG and MG implementations from warp reduction
    // striped_warp_reduction_Implementation<T> striped_warp_reduction_implementation;

    void striped_coloringPrecomputed_computeSymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
    );
};

// we expose the kernel in case we need to call it from another method


#endif // STRIPED_COLORINGPRECOMPUTED_CUH