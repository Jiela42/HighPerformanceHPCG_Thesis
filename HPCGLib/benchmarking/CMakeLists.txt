cmake_minimum_required(VERSION 3.10)

project(benchmarkingLib)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(run_bench_test
    run_bench_test.cpp
)

add_executable(run_full_bench
    run_full_bench.cpp
    abstract_benchmark_functions.cpp
    benchmark_cuSparse.cpp
    benchmark_naiveStriped.cpp
    benchmark_stripedSharedMem.cpp
    benchmark_stripedWarpReduction.cpp
    benchmark_stripedPreprocessed.cpp
    benchmark_stripedColoring.cpp
    benchmark_noStoreStripedColoring.cpp
    benchmark_stripedColoringPrecomputed.cpp
    benchmark_stripedBoxColoring.cpp
    benchmark_stripedCORBoxColoring.cpp

)

add_executable(run_specific_bench
    run_specific_bench.cpp
    abstract_benchmark_functions.cpp
    benchmark_cuSparse.cpp
    benchmark_naiveStriped.cpp
    benchmark_stripedSharedMem.cpp
    benchmark_stripedWarpReduction.cpp
    benchmark_stripedPreprocessed.cpp
    benchmark_stripedColoring.cpp
    benchmark_noStoreStripedColoring.cpp
    benchmark_stripedColoringPrecomputed.cpp
    benchmark_stripedBoxColoring.cpp
    benchmark_stripedCORBoxColoring.cpp
)


target_link_libraries(run_bench_test
    HPCGLib
    TimingLib
)

target_link_libraries(run_full_bench
    HPCGLib
    TimingLib
    TestHPCGLib
    stdc++fs

)

target_link_libraries(run_specific_bench
    HPCGLib
    TimingLib
    TestHPCGLib
    stdc++fs

)