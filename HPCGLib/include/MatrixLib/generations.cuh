#ifndef GENERATIONS_CUH
#define GENERATIONS_CUH

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

#endif // GENERATIONS_CUH