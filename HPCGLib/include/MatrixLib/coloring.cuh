#ifndef COLORING_CUH
#define COLORING_CUH

#include "striped_Matrix.hpp"
#include <cuda_runtime.h>

std::vector<local_int_t> color_for_forward_pass(striped_Matrix<DataType>& A);
std::vector<local_int_t> color_for_backward_pass(striped_Matrix<DataType>& A);
std::pair<std::vector<local_int_t>, std::vector<local_int_t>> get_color_row_mapping(int nx, int ny, int nz);
void get_color_row_mapping(int nx, int ny, int nz, local_int_t * color_pointer_d, local_int_t * color_sorted_row_d);
void get_color_row_mapping_for_boxColoring(int nx, int ny, int nz, local_int_t * color_pointer_d, local_int_t * color_sorted_row_d);
void print_COR_Format(local_int_t max_color, local_int_t num_rows, local_int_t * color_pointer, local_int_t * color_sorted_rows);

__global__ void color_for_forward_pass_kernel(
    local_int_t num_rows, int num_bands, int diag_offset,
    DataType * A, local_int_t * j_min_i, local_int_t * colors
);
__global__ void color_for_backward_pass_kernel(
    local_int_t num_rows, int num_bands, int diag_offset,
    DataType * A, local_int_t * j_min_i, local_int_t * colors
);
__global__ void count_num_rows_per_color_kernel(
    int nx, int ny, int nz, local_int_t * color_pointer
);
__global__ void set_color_pointer_kernel(
    int nx, int ny, int nz, local_int_t * color_pointer
);
__global__ void sort_rows_by_color_kernel(
    int nx, int ny, int nz, local_int_t * color_pointer, local_int_t * color_sorted_rows
);

#endif // COLORING_CUH