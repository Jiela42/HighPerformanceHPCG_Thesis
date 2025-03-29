#ifndef GENERATIONS_HPP
#define GENERATIONS_HPP

#include <vector>
#include <utility>
#include "sparse_CSR_Matrix.hpp"
#include "matrix_basics.hpp"


std::pair<sparse_CSR_Matrix<DataType>, std::vector<DataType>> generate_HPCG_Problem(int nx, int ny, int nz);
std::pair<sparse_CSR_Matrix<DataType>, std::vector<local_int_t>> generate_coarse_HPCG_Problem(int nxf, int nyf, int nzf);

std::vector<DataType> generate_random_vector(int size, int seed);
std::vector<DataType> generate_random_vector(int size, DataType min_val, DataType max_val, int seed);

std::vector<DataType> generate_y_vector_for_HPCG_problem(int nx, int ny, int nz);

#endif // GENERATIONS_HPP