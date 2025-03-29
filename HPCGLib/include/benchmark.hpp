#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP


#include "TimingLib/timer.hpp"
#include "TimingLib/cudaTimer.hpp"
#include "TimingLib/MPITimer.hpp"
#include "testing.hpp"
#include "HPCG_versions/striped_box_coloring.cuh"

#include <string>

// these functions run the benchmark for specific Impelmentations and matrix sizes
void run_cuSparse_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<double>& implementation);
void run_naiveStriped_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, naiveStriped_Implementation<double>& implementation);
void run_stripedSharedMem_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, Striped_Shared_Memory_Implementation<double>& implementation);
void run_striped_warp_reduction_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation);
void run_striped_preprocessed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_preprocessed_Implementation<double>& implementation);
void run_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_coloring_Implementation<double>& implementation);
void run_no_store_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, no_store_striped_coloring_Implementation<double>& implementation);
void run_striped_coloringPrecomputed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_coloringPrecomputed_Implementation<double>& implementation);

void run_cuSparse_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<double>& implementation);
void run_cuSparse_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<double>& implementation);
void run_cuSparse_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<double>& implementation);

void run_warp_reduction_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation);
void run_warp_reduction_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation);
void run_warp_reduction_3d27p_WAXPBY_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation);
void run_warp_reduction_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<double>& implementation);

void run_striped_preprocessed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_preprocessed_Implementation<double>& implementation);

void run_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_coloring_Implementation<double>& implementation);

void run_no_store_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, no_store_striped_coloring_Implementation<double>& implementation);

void run_striped_coloringPrecomputed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_coloringPrecomputed_Implementation<double>& implementation);

void run_striped_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<double>& implementation);
void run_striped_box_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<double>& implementation);
void run_striped_box_coloring_3d27p_CG_benchmark(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<double>& implementation);

void run_striped_COR_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_COR_box_coloring_Implementation<double>& implementation);

void run_multi_GPU_benchmarks(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_multi_GPU_Implementation<DataType>& implementation, Problem *problem, const std::string& benchFilter);

// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<double>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<double> & A,
    double * a_d, double * b_d,
    double * x_d, double * y_d
);

// this version supports striped matrixes
void bench_Implementation(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A, // we need to pass the matrix for potential testing
    double * a_d, double * b_d,
    double * x_d, double * y_d,
    double * result_d,
    double alpha, double beta
    );

// multi GPU version
void bench_Implementation(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    striped_partial_Matrix<DataType> & A, // we need to pass the CSR matrix for metadata and potential testing
    Halo * a_d, Halo * b_d, // a & b are random vectors
    Halo * x_d, Halo * y_d, // x & y are vectors as used in HPCG
    DataType * result_d,  // result is used for the dot product (it is a scalar)
    DataType alpha, DataType beta,
    Problem *problem,
    const std::string& benchFilter
    );

// these functions actually call the functions to be tested
// we have method overloading to support different matrix types
void bench_CG(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d
    );

void bench_CG(
    HPCG_functions<double>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d
    );


void bench_MG(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d
    );

void bench_SPMV(
    HPCG_functions<double>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d
    );

void bench_SPMV(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d
    );

void bench_Dot(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d, double * result_d
    );

void bench_Dot(
    HPCG_functions<double>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d, double * result_d
    );

void bench_WAXPBY(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d, double *w,
    double alpha, double beta
    );


void bench_SymGS(
    HPCG_functions<double>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<double> & A,
    double * x_d, double * y_d
    );

void bench_SymGS(
    HPCG_functions<double>& implementation,
    Timer& timer,
    striped_Matrix<double> & A,
    double * x_d, double * y_d
    );

    
#endif // BENCHMARK_HPP