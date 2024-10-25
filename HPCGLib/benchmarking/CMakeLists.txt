cmake_minimum_required(VERSION 3.10)

project(benchmarkingLib)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bench_test
    bench_test.cpp
)

add_executable(full_bench
    full_bench.cpp
    abstract_benchmark_functions.cpp
    benchmark_cuSparse.cpp
    benchmark_naiveBanded.cpp
)

target_link_libraries(bench_test
    HPCGLib
    TimingLib
)

target_link_libraries(full_bench
    HPCGLib
    TimingLib
    TestHPCGLib
)