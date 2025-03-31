#ifndef UTILS_CUH 
#define UTILS_CUH

#include <MatrixLib/sparse_CSR_Matrix.hpp>
#include <MatrixLib/striped_Matrix.hpp>

int ceiling_division(int numerator, int denominator);
int next_smaller_power_of_two(int n);


__global__ void compute_restriction_kernel(
    local_int_t num_rows,
    DataType * Axf,
    DataType * rf,
    DataType * rc,
    local_int_t* f2c_operator
);

__global__ void compute_restriction_multi_GPU_kernel(
    local_int_t num_rows,
    DataType * Axf,
    DataType * rf,
    DataType * rc,
    local_int_t *f2c_operator,
    int nx, int ny, int nz
);

__global__ void compute_prolongation_kernel(
    local_int_t num_rows,
    DataType * xc,
    DataType * x,
    local_int_t * c2f_operator
);

__global__ void compute_prolongation_multi_GPU_kernel(
    local_int_t num_rows,
    DataType * xc,
    DataType * xf,
    local_int_t * f2c_operator,
    int nx, int ny, int nz
);


double L2_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
);

// double L2_norm_for_SymGS(
//     striped_Matrix<double> & A,
//     double * x,
//     double * y
// );

double relative_residual_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
);

double relative_residual_norm_for_SymGS(
    striped_Matrix<double> & A,
    double * x,
    double * y
);

void L2_norm_for_Device_Vector(
    cudaStream_t stream,
    local_int_t num_rows,
    double * y_d,
    double * L2_norm
);


#endif // UTILS_CUH