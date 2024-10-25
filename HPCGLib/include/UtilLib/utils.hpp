#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>

// Define a constant for error tolerance
const double error_tolerance = 1e-9;

// Function to compare two doubles with a tolerance
bool double_compare(double a, double b);

// Function to compare two vectors of doubles with a tolerance
bool vector_compare(const std::vector<double>& a, const std::vector<double>& b);

#endif // UTILS_HPP