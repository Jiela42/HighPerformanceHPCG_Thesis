cmake_minimum_required(VERSION 3.10)

project(benchmarkingLib)

add_executable(benchmarkingLib
    bench_test.cpp
)

target_link_libraries(benchmarkingLib
    HPCGLib
    TimingLib
)