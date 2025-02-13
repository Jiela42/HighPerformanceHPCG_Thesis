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

#endif // GENERATIONS_CUH