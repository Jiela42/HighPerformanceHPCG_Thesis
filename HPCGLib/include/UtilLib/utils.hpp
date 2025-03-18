#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>
#include <cmath>
// #include "MatrixLib/sparse_CSR_Matrix.hpp"

// Forward declaration of sparse_CSR_Matrix
template <typename T>
class sparse_CSR_Matrix;

// Define a constant for error tolerance
const double error_tolerance = 1e-12;

// Function to compare two doubles with a tolerance
bool double_compare(double a, double b);

bool relaxed_double_compare(double a, double b, double tolerance);

// Function to compare two vectors of doubles with a tolerance
bool vector_compare(const std::vector<double>& a, const std::vector<double>& b);

bool vector_compare(const std::vector<int>& a, const std::vector<int>& b, std::string info);

double L2_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution);

double relative_residual_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution);

void sanity_check_vector( std::vector<double>& a, std::vector<double>& b);
void sanity_check_vectors(std::vector<double *>&device, std::vector<std::vector<double>>& original);

#endif // UTILS_HPP