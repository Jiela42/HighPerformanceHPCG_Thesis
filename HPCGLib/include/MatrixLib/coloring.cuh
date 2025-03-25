#ifndef COLORING_CUH
#define COLORING_CUH

#include "striped_Matrix.hpp"
#include <cuda_runtime.h>

std::vector<int> color_for_forward_pass(striped_Matrix<double>& A);
std::vector<int> color_for_backward_pass(striped_Matrix<double>& A);
std::pair<std::vector<int>, std::vector<int>> get_color_row_mapping(int nx, int ny, int nz);
void get_color_row_mapping(int nx, int ny, int nz, int * color_pointer_d, int * color_sorted_row_d);
void get_color_row_mapping_for_boxColoring(int nx, int ny, int nz, int * color_pointer_d, int * color_sorted_row_d);
void print_COR_Format(int max_color, int num_rows, int * color_pointer, int * color_sorted_rows);

__global__ void color_for_forward_pass_kernel(
    int num_rows, int num_bands, int diag_offset,
    double * A, int * j_min_i, int * colors
);
__global__ void color_for_backward_pass_kernel(
    int num_rows, int num_bands, int diag_offset,
    double * A, int * j_min_i, int * colors
);
__global__ void count_num_rows_per_color_kernel(
    int nx, int ny, int nz, int * color_pointer
);
__global__ void set_color_pointer_kernel(
    int nx, int ny, int nz, int * color_pointer
);
__global__ void sort_rows_by_color_kernel(
    int nx, int ny, int nz, int * color_pointer, int * color_sorted_rows
);

#endif // COLORING_CUH