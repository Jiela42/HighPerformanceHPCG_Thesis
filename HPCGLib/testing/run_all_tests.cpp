#include <iostream>

#include "testing.hpp"

// #include "MatrixLib_tests.cpp"
// #include "HPCG_versions_tests/naiveBanded_tests.cpp"

int main(){
    std::cout << "Starting Full Tests" << std::endl;

    bool all_pass = true;

    // all_pass = all_pass && run_all_matrixLib_tests(4, 4, 4);
    // all_pass = all_pass && run_all_matrixLib_tests(8, 8, 8);
    // all_pass = all_pass && run_all_matrixLib_tests(16, 16, 16);
    // all_pass = all_pass && run_all_matrixLib_tests(32, 32, 32);
    // all_pass = all_pass && run_all_matrixLib_tests(64, 64, 64);
    // all_pass = all_pass && run_all_matrixLib_tests(128, 128, 128);
    std::cout << "Finished MatrixLib tests" << std::endl;

    std::cout<< "Starting cuSparse tests" << std::endl;
    all_pass = all_pass && run_cuSparse_tests(4, 4, 4);
    std::cout << "Finished cuSparse tests" << std::endl;

    std::cout << "Starting naiveBanded tests" << std::endl;
    // all_pass = all_pass && run_naiveBanded_tests(4, 4, 4);
    // all_pass = all_pass && run_naiveBanded_tests(8, 8, 8);
    // all_pass = all_pass && run_naiveBanded_tests(16, 16, 16);
    // all_pass = all_pass && run_naiveBanded_tests(32, 32, 32);
    // all_pass = all_pass && run_naiveBanded_tests(64, 64, 64);
    // all_pass = all_pass && run_naiveBanded_tests(128, 128, 128);
    std::cout << "Finished naiveBanded tests" << std::endl;

    std::cout << "Starting banded shared memory tests" << std::endl;
    // all_pass = all_pass && run_bandedSharedMem_tests(4, 4, 4);
    // all_pass = all_pass && run_bandedSharedMem_tests(8, 8, 8);
    // all_pass = all_pass && run_bandedSharedMem_tests(16, 16, 16);
    // all_pass = all_pass && run_bandedSharedMem_tests(32, 32, 32);
    // all_pass = all_pass && run_bandedSharedMem_tests(64, 64, 64);
    // all_pass = all_pass && run_bandedSharedMem_tests(128, 128, 128);
    std::cout << "Finished banded shared memory tests" << std::endl;

    std::cout << "Starting banded warp reduction tests" << std::endl;
    // all_pass = all_pass && run_bandedWarpReduction_tests(4, 4, 4);
    // all_pass = all_pass && run_bandedWarpReduction_tests(8, 8, 8);
    // all_pass = all_pass && run_bandedWarpReduction_tests(16, 16, 16);
    // all_pass = all_pass && run_bandedWarpReduction_tests(24, 24, 24);
    // all_pass = all_pass && run_bandedWarpReduction_tests(32, 32, 32);
    // all_pass = all_pass && run_bandedWarpReduction_tests(64, 64, 64);
    // all_pass = all_pass && run_bandedWarpReduction_tests(128, 128, 128);
    std::cout << "Finished banded warp reduction tests" << std::endl;


    // std::cout << "Starting banded preprocessed tests" << std::endl;
    // all_pass = all_pass && run_bandedPreprocessed_tests(4, 4, 4);
    // all_pass = all_pass && run_bandedPreprocessed_tests(8, 8, 8);
    // all_pass = all_pass && run_bandedPreprocessed_tests(16, 16, 16);
    // all_pass = all_pass && run_bandedPreprocessed_tests(32, 32, 32);
    // all_pass = all_pass && run_bandedPreprocessed_tests(64, 64, 64);

    // std::cout << "Finished banded preprocessed tests" << std::endl;

    std::cout << "Starting banded colored tests" << std::endl;
    all_pass = all_pass && run_bandedColored_tests(4, 4, 4);
    all_pass = all_pass && run_bandedColored_tests(8, 8, 8);
    all_pass = all_pass && run_bandedColored_tests(16, 16, 16);
    all_pass = all_pass && run_bandedColored_tests(32, 32, 32);
    all_pass = all_pass && run_bandedColored_tests(64, 64, 64);
    all_pass = all_pass && run_bandedColored_tests(128, 128, 128);

    if (all_pass){
        std::cout << "*******************************************************************************************" << std::endl;
        std::cout << "********************************FINISHED & PASSED ALL TESTS********************************" << std::endl;
        std::cout << "*******************************************************************************************" << std::endl;

    } else {
        std::cout << "Some tests failed -> Go do debugging" << std::endl;
    }
    
}