cmake_minimum_required(VERSION 3.10)

project(TestHPCGLib)

add_executable(TestHPCGLib
    quicktesting.cpp
)

target_link_libraries(TestHPCGLib
    HPCGLib
    MatrixLib
)