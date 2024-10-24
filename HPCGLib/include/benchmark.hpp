#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include "HPCGLib.hpp"
#include "HPCG_versions/cusparse.hpp"
#include "HPCG_versions/naiveBanded.cuh"

#include "TimingLib/timer.hpp"
#include "testing.hpp"

#include <string>

// these functions run the benchmark for specific Impelmentations and matrix sizes
void run_cuSparse_3d27p__benchmarks(int nx, int ny, int nz, std::string folder_path);


// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    const sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d);

// this version supports banded matrixes
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    const sparse_CSR_Matrix<double> & A, // we need to pass the CSR matrix for potential testing
    double * banded_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * x_d, double * y_d
    );


// these functions actually call the functions to be tested
// we have method overloading to support different matrix types
void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    const sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d);

void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    double * banded_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * x_d, double * y_d
    );
#endif // BENCHMARK_HPP