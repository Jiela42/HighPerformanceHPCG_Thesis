cmake_minimum_required(VERSION 3.10)

project(Testing LANGUAGES CXX CUDA)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)

add_executable(TestHPCGLib
    quicktesting.cpp
)

add_executable(AllTests
    all_tests.cpp

    HPCG_versions_tests/naiveBanded_tests.cpp
    
    MatrixLib_tests.cpp
    HPCG_versions_tests/test_calls.cpp

)

target_link_libraries(TestHPCGLib
    HPCGLib
    MatrixLib
)

target_link_libraries(AllTests
    HPCGLib
    MatrixLib
)