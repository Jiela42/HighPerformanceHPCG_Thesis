#ifndef GENERATIONS_HPP
#define GENERATIONS_HPP

#include <vector>
#include <utility>
#include "sparse_CSR_Matrix.hpp"
#include "matrix_basics.hpp"


std::pair<sparse_CSR_Matrix<double>, std::vector<double>> generate_HPCG_Problem(int nx, int ny, int nz);
std::pair<sparse_CSR_Matrix<double>, std::vector<int>> generate_coarse_HPCG_Problem(int nxf, int nyf, int nzf);

std::vector<double> generate_random_vector(int size, int seed);
std::vector<double> generate_random_vector(int size, double min_val, double max_val, int seed);

std::vector<double> generate_y_vector_for_HPCG_problem(int nx, int ny, int nz);

#endif // GENERATIONS_HPP