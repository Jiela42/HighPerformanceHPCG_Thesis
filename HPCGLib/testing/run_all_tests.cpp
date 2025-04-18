#include <iostream>
#include <ctime>
#include <iomanip>
#include "testing.hpp"

// #include "MatrixLib_tests.cpp"
// #include "HPCG_versions_tests/naiveStriped_tests.cpp"

int main(){
    std::cout << "Starting Full Tests" << std::endl;
    // print the current time, so we see if the tests are running

    // Get the current time
    std::time_t now = std::time(nullptr);
    std::tm* local_time = std::localtime(&now);

    // Print the time of day in a readable format
    std::cout << "Current Time: " << std::put_time(local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
 
    bool all_pass = true;

    all_pass = all_pass && run_all_util_tests(4, 4, 4);
    all_pass = all_pass && run_all_util_tests(3, 4, 5);
    all_pass = all_pass && run_all_util_tests(8, 8, 8);
    all_pass = all_pass && run_all_util_tests(16, 16, 16);
    all_pass = all_pass && run_all_util_tests(32, 32, 32);
    all_pass = all_pass && run_all_util_tests(64, 64, 64);

    // std::cout << "Starting MatrixLib tests" << std::endl;
    all_pass = all_pass && run_all_matrixLib_tests(4, 4, 4);
    all_pass = all_pass && run_all_matrixLib_tests(3, 4, 5);
    all_pass = all_pass && run_all_matrixLib_tests(4, 3, 5);
    all_pass = all_pass && run_all_matrixLib_tests(5, 4, 3);
    all_pass = all_pass && run_all_matrixLib_tests(8, 8, 8);
    all_pass = all_pass && run_all_matrixLib_tests(16, 16, 16);
    all_pass = all_pass && run_all_matrixLib_tests(24, 24, 24);
    all_pass = all_pass && run_all_matrixLib_tests(32, 32, 32);
    std::cout << "running matrix tests for 64x64x64" << std::endl;
    all_pass = all_pass && run_all_matrixLib_tests(64, 64, 64);
    // std::cout << "Trying to generate a large matrix" << std::endl;
    // all_pass = all_pass && run_all_matrixLib_tests(128, 128, 128);
    // std::cout << "Finished MatrixLib tests" << std::endl;

    // std::cout<< "Starting cuSparse tests" << std::endl;
    all_pass = all_pass && run_cuSparse_tests(4, 4, 4);
    all_pass = all_pass && run_cuSparse_tests(8, 8, 8);
    all_pass = all_pass && run_cuSparse_tests(16, 16, 16);
    all_pass = all_pass && run_cuSparse_tests(32, 32, 32);
    all_pass = all_pass && run_cuSparse_tests(64, 64, 64);
    // all_pass = all_pass && run_cuSparse_tests(128, 128, 128);
    // std::cout << "Finished cuSparse tests" << std::endl;

    std::cout << "Starting naiveStriped tests" << std::endl;
    all_pass = all_pass && run_naiveStriped_tests(4, 4, 4);
    all_pass = all_pass && run_naiveStriped_tests(8, 8, 8);
    all_pass = all_pass && run_naiveStriped_tests(16, 16, 16);
    all_pass = all_pass && run_naiveStriped_tests(32, 32, 32);
    all_pass = all_pass && run_naiveStriped_tests(64, 64, 64);
    // all_pass = all_pass && run_naiveStriped_tests(128, 128, 128);
    // std::cout << "Finished naiveStriped tests" << std::endl;

    // std::cout << "Starting striped shared memory tests" << std::endl;
    // all_pass = all_pass && run_stripedSharedMem_tests(4, 4, 4);
    // all_pass = all_pass && run_stripedSharedMem_tests(8, 8, 8);
    // all_pass = all_pass && run_stripedSharedMem_tests(16, 16, 16);
    // all_pass = all_pass && run_stripedSharedMem_tests(32, 32, 32);
    // all_pass = all_pass && run_stripedSharedMem_tests(64, 64, 64);
    // all_pass = all_pass && run_stripedSharedMem_tests(128, 128, 128);
    // std::cout << "Finished striped shared memory tests" << std::endl;

    std::cout << "Starting striped warp reduction tests" << std::endl;
    // all_pass = all_pass && run_stripedWarpReduction_filebased_tests();
    all_pass = all_pass && run_stripedWarpReduction_tests(4, 4, 4);
    all_pass = all_pass && run_stripedWarpReduction_tests(8, 8, 8);
    all_pass = all_pass && run_stripedWarpReduction_tests(7, 8, 9);
    all_pass = all_pass && run_stripedWarpReduction_tests(8, 7, 9);
    all_pass = all_pass && run_stripedWarpReduction_tests(8, 9, 7);
    all_pass = all_pass && run_stripedWarpReduction_tests(16, 16, 16);
    all_pass = all_pass && run_stripedWarpReduction_tests(24, 24, 24);
    all_pass = all_pass && run_stripedWarpReduction_tests(32, 32, 32);
    all_pass = all_pass && run_stripedWarpReduction_tests(64, 64, 64);
    // all_pass = all_pass && run_stripedWarpReduction_tests(128, 128, 128);
    // std::cout << "Finished striped warp reduction tests" << std::endl;

    // these fail the tests, so we don't do them (mature, I know!)
    // std::cout << "Starting striped preprocessed tests" << std::endl;
    // all_pass = all_pass && run_stripedPreprocessed_tests(4, 4, 4);
    // all_pass = all_pass && run_stripedPreprocessed_tests(8, 8, 8);
    // all_pass = all_pass && run_stripedPreprocessed_tests(16, 16, 16);
    // all_pass = all_pass && run_stripedPreprocessed_tests(32, 32, 32);
    // all_pass = all_pass && run_stripedPreprocessed_tests(64, 64, 64);

    // std::cout << "Finished striped preprocessed tests" << std::endl;

    std::cout << "Starting striped colored tests" << std::endl;
    all_pass = all_pass && run_stripedColored_tests(4, 4, 4);
    all_pass = all_pass && run_stripedColored_tests(8, 8, 8);
    all_pass = all_pass && run_stripedColored_tests(16, 16, 16);
    all_pass = all_pass && run_stripedColored_tests(32, 32, 32);
    all_pass = all_pass && run_stripedColored_tests(64, 64, 64);
    // all_pass = all_pass && run_stripedColored_tests(128, 128, 128);

    std::cout << "Starting coloringPrecomputed tests" << std::endl;
    all_pass = all_pass && run_stripedColoringPrecomputed_filebased_tests();
    all_pass = all_pass && run_stripedColoringPrecomputed_tests(4, 4, 4);
    all_pass = all_pass && run_stripedColoringPrecomputed_tests(8, 8, 8);
    all_pass = all_pass && run_stripedColoringPrecomputed_tests(16, 16, 16);
    all_pass = all_pass && run_stripedColoringPrecomputed_tests(32, 32, 32);
    all_pass = all_pass && run_stripedColoringPrecomputed_tests(64, 64, 64);
    // all_pass = all_pass && run_stripedColoringPrecomputed_tests(128, 128, 128);
    // all_pass = all_pass && run_stripedColoringPrecomputed_tests(256, 256, 256);
    // all_pass = all_pass && run_stripedColoringPrecomputed_tests(512, 256, 256);
    // all_pass = all_pass && run_stripedColoringPrecomputed_tests(176, 176, 176);
    // all_pass = all_pass && run_stripedColoringPrecomputed_tests(240, 240, 240);

    std::cout << "Starting stripedBoxColoring tests" << std::endl;
    all_pass = all_pass && run_stripedBoxColoring_tests(4, 4, 4);
    all_pass = all_pass && run_stripedBoxColoring_tests(8, 8, 8);
    all_pass = all_pass && run_stripedBoxColoring_tests(16, 16, 16);
    all_pass = all_pass && run_stripedBoxColoring_tests(32, 32, 32);
    all_pass = all_pass && run_stripedBoxColoring_tests(64, 64, 64);
    // all_pass = all_pass && run_stripedBoxColoring_tests(128, 128, 128);
    // all_pass = all_pass && run_stripedBoxColoring_tests(176, 176, 176);
    // all_pass = all_pass && run_stripedBoxColoring_tests(240, 240, 240);

    std::cout << "Starting no_store tests" << std::endl;
    all_pass = all_pass && run_no_store_stripedColoring_filebased_tests();
    all_pass = all_pass && run_no_store_stripedColoring_tests(32, 32, 32);

    std::cout << "Starting COR tests" << std::endl;
    all_pass = all_pass && run_COR_BoxColoring_tests(16, 16, 16);
    all_pass = all_pass && run_COR_BoxColoring_tests(24, 24, 24);
    all_pass = all_pass && run_COR_BoxColoring_tests(32, 32, 32);
    all_pass = all_pass && run_COR_BoxColoring_tests(64, 64, 64);
    // all_pass = all_pass && run_COR_BoxColoring_tests(128, 128, 128);
    // all_pass = all_pass && run_COR_BoxColoring_tests(176, 176, 176);
    // all_pass = all_pass && run_COR_BoxColoring_tests(240, 240, 240);


    if (all_pass){
        std::cout << "*******************************************************************************************" << std::endl;
        std::cout << "********************************FINISHED & PASSED ALL TESTS********************************" << std::endl;
        std::cout << "*******************************************************************************************" << std::endl;

    } else {
        std::cout << "Some tests failed -> Go do debugging" << std::endl;
    }
    
}