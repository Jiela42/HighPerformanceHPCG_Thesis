#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP


#include "TimingLib/timer.hpp"
#include "testing.hpp"
#include "HPCG_versions/striped_box_coloring.cuh"

#include <string>

// these functions run the benchmark for specific Impelmentations and matrix sizes
void run_cuSparse_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_naiveStriped_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_stripedSharedMem_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_striped_warp_reduction_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_striped_preprocessed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_no_store_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);
void run_striped_coloringPrecomputed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);

void run_cuSparse_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path);
void run_cuSparse_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path);
void run_cuSparse_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path);

void run_warp_reduction_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path);
void run_warp_reduction_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path);
void run_warp_reduction_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path);

void run_striped_preprocessed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path);

void run_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path);

void run_no_store_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path);

void run_striped_coloringPrecomputed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path);

void run_striped_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path);

// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * a_d, double * b_d,
    double * x_d, double * y_d);

// this version supports striped matrixes
void bench_Implementation(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A, // we need to pass the CSR matrix for potential testing
    double * striped_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * a_d, double * b_d,
    double * x_d, double * y_d,
    double * result_d
    );


// these functions actually call the functions to be tested
// we have method overloading to support different matrix types
void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d, double * y_d
    );

void bench_SPMV(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A, // we pass A for the metadata and for testing against cuSparse
    double * striped_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * x_d, double * y_d
    );

void bench_Dot(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d, double * result_d
    );

void bench_Dot(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d, double * result_d
    );

void bench_SymGS(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    sparse_CSR_Matrix<double> & A,
    int * A_row_ptr_d, int * A_col_idx_d, double * A_values_d,
    double * x_d, double * y_d
    );

void bench_SymGS(
    HPCG_functions<double>& implementation,
    CudaTimer& timer,
    striped_Matrix<double> & A, // we pass A for the metadata and for testing against cuSparse
    double * striped_A_d,
    int num_rows, int num_cols,
    int num_bands,
    int * j_min_i_d,
    double * x_d, double * y_d
    );

    
#endif // BENCHMARK_HPP