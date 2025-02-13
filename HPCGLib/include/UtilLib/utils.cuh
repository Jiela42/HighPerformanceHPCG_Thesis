#ifndef UTILS_CUH 
#define UTILS_CUH

#include <MatrixLib/sparse_CSR_Matrix.hpp>
#include <MatrixLib/striped_Matrix.hpp>

int ceiling_division(int numerator, int denominator);
int next_smaller_power_of_two(int n);

double L2_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
);

double L2_norm_for_SymGS(
    striped_Matrix<double> & A,
    double * x,
    double * y
);

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
    int num_rows,
    double * y_d,
    double * L2_norm
);


#endif // UTILS_CUH