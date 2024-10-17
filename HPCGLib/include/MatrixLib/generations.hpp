#ifndef GENERATIONS_HPP
#define GENERATIONS_HPP

#include <vector>
#include <utility>
#include "sparse_CSR_Matrix.hpp"


std::pair<sparse_CSR_Matrix<double>, std::vector<double>> generate_HPCG_Problem(int nx, int ny, int nz);
std::pair<sparse_CSR_Matrix<double>, std::vector<int>> generate_coarse_HPCG_Problem(int nxf, int nyf, int nzf);


#endif // GENERATIONS_HPP