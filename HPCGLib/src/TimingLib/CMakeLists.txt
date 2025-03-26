# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

add_library(TimingLib)

target_sources(TimingLib PRIVATE
    timer.cpp
    cudaTimer.cpp
    cpuTimer.cpp
)

target_link_libraries(TimingLib
    HPCGLib
)