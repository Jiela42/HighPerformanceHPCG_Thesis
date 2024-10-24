#include <iostream>

#include "testing.hpp"

// #include "MatrixLib_tests.cpp"
// #include "HPCG_versions_tests/naiveBanded_tests.cpp"

int main(){
    std::cout << "Starting Full Tests" << std::endl;

    run_all_MatrixLib_tests(8, 8, 8);
    run_all_MatrixLib_tests(16, 16, 16);
    run_all_MatrixLib_tests(32, 32, 32);
    std::cout << "Finished MatrixLib tests" << std::endl;

    std::cout << "Starting naiveBanded tests" << std::endl;
    run_naiveBanded_tests(8, 8, 8);
    run_naiveBanded_tests(16, 16, 16);
    run_naiveBanded_tests(32, 32, 32);
    std::cout << "Finished naiveBanded tests" << std::endl;


    std::cout << "*******************************************************************************************" << std::endl;
    std::cout << "********************************FINISHED & PASSED ALL TESTS********************************" << std::endl;
    std::cout << "*******************************************************************************************" << std::endl;

}