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
    benchmark_naiveBanded.cpp
    benchmark_bandedSharedMem.cpp
    benchmark_bandedWarpReduction.cpp
)

target_link_libraries(run_bench_test
    HPCGLib
    TimingLib
)

target_link_libraries(run_full_bench
    HPCGLib
    TimingLib
    TestHPCGLib
)