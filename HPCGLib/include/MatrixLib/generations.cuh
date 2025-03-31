#ifndef GENERATIONS_CUH
#define GENERATIONS_CUH

#include "UtilLib/hpcg_multi_GPU_utils.cuh"

void generateHPCGProblem(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values,
    DataType * y    
);

void generateHPCGMatrix(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values
);

local_int_t generate_striped_3D27P_Matrix_from_CSR(
    int nx, int ny, int nz,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values,
    int num_stripes, local_int_t * j_min_i,
    DataType * striped_A_d
);

local_int_t generate_CSR_from_Striped(
    local_int_t num_rows, int num_stripes,
    local_int_t * j_min_i, DataType * striped_A_d,
    local_int_t * row_ptr, local_int_t * col_idx, DataType * values
);

void generate_f2c_operator(
    int nxf, int nyf, int nzf,
    int nxc, int nyc, int nzc,
    local_int_t * f2c_operator
);

void GenerateStripedPartialMatrix_GPU(
    Problem *problem, 
    DataType *A_d
);

void generate_partialf2c_operator(
    int nxc, int nyc, int nzc,
    int nxf, int nyf, int nzf,
    local_int_t * f2c_op_d
);

void generate_y_vector_for_HPCG_problem_onGPU(
    Problem *problem, 
    DataType *y_d);

#endif // GENERATIONS_CUH