#ifndef GENERATIONS_HPP
#define GENERATIONS_HPP

#include <vector>
#include <utility>
#include "sparse_CSR_Matrix.hpp"
#include "matrix_basics.hpp"


std::pair<sparse_CSR_Matrix<DataType>, std::vector<DataType>> generate_HPCG_Problem(int nx, int ny, int nz);
std::pair<sparse_CSR_Matrix<DataType>, std::vector<local_int_t>> generate_coarse_HPCG_Problem(int nxf, int nyf, int nzf);

std::vector<DataType> generate_random_vector(global_int_t size, int seed);
std::vector<DataType> generate_random_vector(global_int_t size, DataType min_val, DataType max_val, int seed);

std::vector<DataType> generate_y_vector_for_HPCG_problem(int nx, int ny, int nz);

global_int_t pick_random(global_int_t num_values);
void GenerateRandomCOOPartialMatrix_CPU(global_int_t *row, global_int_t *col, DataType *data, Problem *p, global_int_t nnz);

#endif // GENERATIONS_HPP