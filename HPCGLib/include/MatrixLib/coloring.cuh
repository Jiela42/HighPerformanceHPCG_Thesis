#ifndef COLORING_CUH
#define COLORING_CUH

#include "banded_Matrix.hpp"
#include <cuda_runtime.h>

std::vector<int> color_for_forward_pass(banded_Matrix<double> A);
__global__ void color_for_forward_pass_kernel(
    int num_rows, int num_bands, int diag_offset, double * A, int * j_min_i, int * colors
);

#endif // COLORING_CUH