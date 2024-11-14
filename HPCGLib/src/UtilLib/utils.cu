#include "UtilLib/utils.cuh"

#include <cmath>

int ceiling_division(int numerator, int denominator) {
    return static_cast<int>(std::ceil(static_cast<double>(numerator) / denominator));
}

int next_smaller_power_of_two(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power / 2;
}