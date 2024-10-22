#include "utils.cuh"

#include <cmath>

int ceiling_division(int numerator, int denominator) {
    return static_cast<int>(std::ceil(static_cast<double>(numerator) / denominator));
}