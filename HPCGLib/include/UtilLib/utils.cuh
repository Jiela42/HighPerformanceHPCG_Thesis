#ifndef UTILS_CUH 
#define UTILS_CUH

int ceiling_division(int numerator, int denominator);
int next_smaller_power_of_two(int n);

double L2_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int * row_ptr,
    int * col_idx,
    double * values,
    double * x,
    double * y
);

double L2_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int num_stripes,
    int * j_min_i,
    double * A,
    double * x,
    double * y
);

double relative_residual_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int * row_ptr,
    int * col_idx,
    double * values,
    double * x,
    double * y
);

double relative_residual_norm_for_SymGS(
    int num_rows,
    int num_cols,
    int num_stripes,
    int * j_min_i,
    double * A,
    double * x,
    double * y
);

#endif // UTILS_CUH