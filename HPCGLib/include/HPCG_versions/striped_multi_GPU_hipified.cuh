#include "hip/hip_runtime.h"
#ifndef STRIPED_MULTI_GPU_CUH
#define STRIPED_MULTI_GPU_CUH

#include "HPCGLib_hipified.hpp"
#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"
#include "MatrixLib/striped_Matrix_hipified.hpp"
#include "MatrixLib/striped_partial_Matrix_hipified.hpp"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"

#include <hip/hip_runtime.h>

template <typename T>
class striped_multi_GPU_Implementation : public HPCG_functions<T> {
public:

    int dot_cooperation_number; // the cooperation number for the dot product
    int bx, by, bz; // the box size in x, y and z direction
    int SymGS_cooperation_number; // the cooperation number for the SymGS

    striped_multi_GPU_Implementation(){
        // overwritting the inherited variables

        this->dot_cooperation_number = 8;

        this->doPreconditioning = false;

        this->version_name = "Striped Multi GPU";
        this->implementation_type = Implementation_Type::STRIPED;
        this->SPMV_implemented = true;
        this->Dot_implemented = true;
        this->SymGS_implemented = true;
        this->WAXPBY_implemented = true;
        this->CG_implemented = true;
        this->MG_implemented = true;
        this->norm_based = false;

        // default box size for coloring
        this->bx = 3;
        this->by = 3;
        this->bz = 3;

        // set the default cooperation number for the SymGS
        this->SymGS_cooperation_number = 4;

    }

    virtual Problem* init_comm(
        int argc, char *argv[],
        int npx, int npy, int npz,
        int nx, int ny, int nz
        ) = 0;

    virtual void ExchangeHalo(
        Halo *halo, Problem *problem
    ) = 0;
        
    virtual void finalize_comm(
        Problem *problem
    ) = 0;

    void compute_CG(
        striped_Matrix<T> & A,
        T * b_d, T * x_d,
        int & n_iters, T& normr, T& normr0
    ) override {
        std::cerr << "Warning: compute_CG requires different arguments in multi GPU." << std::endl;
    }

    void compute_CG(
        striped_partial_Matrix<T> & A,
        Halo * b_d, Halo * x_d,
        int & n_iters, T& normr, T& normr0,
        Problem *problem
    ) {
        striped_warp_reduction_multi_GPU_computeCG(A, b_d, x_d, n_iters, normr, normr0, problem);
    }

    void compute_MG(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
            std::cerr << "Warning: compute_MG requires different arguments in multi GPU." << std::endl;
    }
    
    void compute_MG(
        striped_partial_Matrix<T> & A,
        Halo * x_d, Halo * y_d, // the vectors x and y are already on the device
        Problem *problem
        ) {
            striped_warp_reduction_multi_GPU_computeMG(A, x_d, y_d, problem);
    }

    void compute_SymGS(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in multi GPU." << std::endl;
    }

    void compute_SymGS(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        std::cerr << "Warning: compute_SymGS requires different arguments in multi GPU." << std::endl;
    }

    void compute_SymGS(
        striped_partial_Matrix<T> & A,
        Halo * x_d, Halo * b_d, // the vectors x and y are already on the device
        Problem *problem
    ) {
        striped_box_coloring_multi_GPU_computeSymGS(A, x_d, b_d, problem);
    }
    
    void compute_SPMV(
        sparse_CSR_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires a striped Matrix as input for multi GPU Implementation.");
    }

    void compute_SPMV(
        striped_Matrix<T> & A,
        T * x_d, T * y_d // the vectors x and y are already on the device
        ) override {
        throw std::runtime_error("ERROR: compute_SPMV requires different arguments as input for multi GPU Implementation.");
    }

    void compute_SPMV(
        striped_partial_Matrix<T>& A,
        Halo * x_d, Halo * b_d, // the vectors x and y are already on the device
        Problem *problem
        ) {
        striped_warp_reduction_multi_GPU_computeSPMV(A, x_d, b_d, problem);
    }

    void compute_WAXPBY(
        sparse_CSR_Matrix<T>& A, // we pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "ERROR computeWAXPBY requires different arguments as input for multi GPU Implementation." << std::endl;
    }

    void compute_WAXPBY(
        striped_Matrix<T>& A, // we pass A for the metadata
        T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta
        ) override {
        std::cerr << "ERROR computeWAXPBY requires different arguments as input for multi GPU Implementation." << std::endl;
    }

    void compute_WAXPBY(
        Halo * x_d, Halo * y_d, Halo * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta,
        Problem *problem,
        bool updateHalo
        ) {
        striped_warp_reduction_multi_GPU_computeWAXPBY(x_d, y_d, w_d, alpha, beta, problem, updateHalo);
    }

    void compute_Dot(
        sparse_CSR_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
        std::cerr << "ERROR computeDot requires different arguments as input for multi GPU Implementation." << std::endl;
    }

    void compute_Dot(
        striped_Matrix<T> & A, // we pass A for the metadata
        T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) override {
            std::cerr << "ERROR computeDot requires different arguments as input for multi GPU Implementation." << std::endl;
    }

    void compute_Dot(
        Halo * x_d, Halo * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) {
        striped_warp_reduction_multi_GPU_computeDot(x_d, y_d, result_d);
    }

    void set_box_size(int bx, int by, int bz){
        this->bx = bx;
        this->by = by;
        this->bz = bz;
    }

private:

    void striped_warp_reduction_multi_GPU_computeCG(
        striped_partial_Matrix<T> & A,
        Halo * b_d, Halo * x_d,
        int & n_iters, T& normr, T& normr0,
        Problem *problem
    );

    void striped_warp_reduction_multi_GPU_computeMG(
        striped_partial_Matrix<T> & A,
        Halo * x_d, Halo * y_d, // the vectors x and y are already on the device
        Problem *problem
    );

    void striped_warp_reduction_multi_GPU_computeSPMV(
        striped_partial_Matrix<T>& A,
        Halo *x_d, Halo *y_d, // the vectors x and y are already on the device
        Problem *problem
    );

    void striped_warp_reduction_multi_GPU_computeDot(
        Halo * x_d, Halo * y_d, T * result_d
    );

    void striped_box_coloring_multi_GPU_computeSymGS(
        striped_partial_Matrix<T> & A,
        Halo *x_d, Halo *b_d, // the vectors x and y are already on the device
        Problem *problem
    );

    void striped_warp_reduction_multi_GPU_computeWAXPBY(
        Halo * x_d, Halo * y_d, Halo * w_d, // the vectors x, y and w are already on the device
        T alpha, T beta,
        Problem *problem,
        bool updateHalo
    );
};

// we expose the kernel in case we need to call it from another method
__global__ void striped_warp_reduction_multi_GPU_SPMV_kernel(
    double* striped_A,
    local_int_t num_rows, int num_stripes, local_int_t * j_min_i,
    double* x, double* y, int nx, int ny, int nz, 
    global_int_t gnx, global_int_t gny, global_int_t gnz, 
    global_int_t gi0,
    int px, int py, int pz
    );
__global__ void striped_warp_reduction_multi_GPU_dot_kernel(
    local_int_t num_rows,
    double * x_d,
    double * y_d,
    double * result_d,
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
);


#endif // STRIPED_MULTI_GPU_CUH