
add_executable(run_bench_test
    run_bench_test_hipified.cpp
)

add_executable(run_full_bench
    run_full_bench_hipified.cpp
    abstract_benchmark_functions_hipified.cpp
    benchmark_cuSparse_hipified.cpp
    benchmark_naiveStriped_hipified.cpp
    benchmark_stripedSharedMem_hipified.cpp
    benchmark_stripedWarpReduction_hipified.cpp
    benchmark_stripedPreprocessed_hipified.cpp
    benchmark_stripedColoring_hipified.cpp
    benchmark_noStoreStripedColoring_hipified.cpp
    benchmark_stripedColoringPrecomputed_hipified.cpp
    benchmark_stripedBoxColoring_hipified.cpp
    benchmark_stripedCORBoxColoring_hipified.cpp

)

add_executable(run_specific_bench
    run_specific_bench_hipified.cpp
    abstract_benchmark_functions_hipified.cpp
    benchmark_cuSparse_hipified.cpp
    benchmark_naiveStriped_hipified.cpp
    benchmark_stripedSharedMem_hipified.cpp
    benchmark_stripedWarpReduction_hipified.cpp
    benchmark_stripedPreprocessed_hipified.cpp
    benchmark_stripedColoring_hipified.cpp
    benchmark_noStoreStripedColoring_hipified.cpp
    benchmark_stripedColoringPrecomputed_hipified.cpp
    benchmark_stripedBoxColoring_hipified.cpp
    benchmark_stripedCORBoxColoring_hipified.cpp
)

add_executable(run_maxSizeSingleGPU_finder
    run_maxSizeSingleGPU_finder_hipified.cpp
    abstract_benchmark_functions_hipified.cpp
    benchmark_stripedColoring_hipified.cpp
    benchmark_stripedBoxColoring_hipified.cpp
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

target_link_libraries(run_maxSizeSingleGPU_finder
    HPCGLib
    TimingLib
    TestHPCGLib
    stdc++fs
)