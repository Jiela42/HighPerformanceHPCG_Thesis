#ifndef GENERATIONS_CUH
#define GENERATIONS_CUH

#include "UtilLib/hpcg_multi_GPU_utils.cuh"

void generateHPCGProblem(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values,
    double * y    
);

void generateHPCGMatrix(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values
);

int generate_striped_3D27P_Matrix_from_CSR(
    int nx, int ny, int nz,
    int * row_ptr, int * col_idx, double * values,
    int num_stripes, int * j_min_i,
    double * striped_A_d
);

int generate_CSR_from_Striped(
    int num_rows, int num_stripes,
    int * j_min_i, double * striped_A_d,
    int * row_ptr, int * col_idx, double * values
);

void generate_f2c_operator(
    int nxf, int nyf, int nzf,
    int nxc, int nyc, int nzc,
    int * f2c_operator
);

void GenerateStripedPartialMatrix_GPU(
    Problem *problem, 
    double *A_d
);

void generate_partialf2c_operator(
    int nxc, int nyc, int nzc,
    int nxf, int nyf, int nzf,
    int * f2c_op_d
);

void generate_y_vector_for_HPCG_problem_onGPU(
    Problem *problem, 
    double *y_d);

#endif // GENERATIONS_CUH