#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP


#include "TimingLib/timer_hipified.hpp"
#include "TimingLib/cudaTimer_hipified.hpp"
#include "TimingLib/MPITimer_hipified.hpp"
#include "testing_hipified.hpp"
#include "HPCG_versions/striped_box_coloring_hipified.cuh"

#include <string>

// these functions run the benchmark for specific Impelmentations and matrix sizes
void run_cuSparse_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation);
void run_naiveStriped_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, naiveStriped_Implementation<DataType>& implementation);
void run_stripedSharedMem_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, Striped_Shared_Memory_Implementation<DataType>& implementation);
void run_striped_warp_reduction_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation);
void run_striped_preprocessed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_preprocessed_Implementation<DataType>& implementation);
void run_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_coloring_Implementation<DataType>& implementation);
void run_no_store_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, no_store_striped_coloring_Implementation<DataType>& implementation);
void run_striped_coloringPrecomputed_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_coloringPrecomputed_Implementation<DataType>& implementation);

void run_cuSparse_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation);
void run_cuSparse_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation);
void run_cuSparse_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, cuSparse_Implementation<DataType>& implementation);

void run_warp_reduction_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation);
void run_warp_reduction_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation);
void run_warp_reduction_3d27p_WAXPBY_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation);
void run_warp_reduction_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation);

void run_striped_preprocessed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_preprocessed_Implementation<DataType>& implementation);

void run_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_coloring_Implementation<DataType>& implementation);

void run_no_store_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, no_store_striped_coloring_Implementation<DataType>& implementation);

void run_striped_coloringPrecomputed_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_coloringPrecomputed_Implementation<DataType>& implementation);

void run_striped_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<DataType>& implementation);
void run_striped_box_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<DataType>& implementation);
void run_striped_box_coloring_3d27p_CG_benchmark(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<DataType>& implementation);

void run_striped_COR_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_COR_box_coloring_Implementation<DataType>& implementation);

void run_multi_GPU_benchmarks(int npx, int npy, int npz, int nx, int ny, int nz, std::string folder_path, striped_multi_GPU_Implementation<DataType>& implementation, Problem *problem, const std::string& benchFilter);

// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * a_d, DataType * b_d,
    DataType * x_d, DataType * y_d
);

// this version supports striped matrixes
void bench_Implementation(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A, // we need to pass the matrix for potential testing
    DataType * a_d, DataType * b_d,
    DataType * x_d, DataType * y_d,
    DataType * result_d,
    DataType alpha, DataType beta
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
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );

void bench_CG(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );


void bench_MG(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );

void bench_SPMV(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );

void bench_SPMV(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );

void bench_Dot(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d, DataType * result_d
    );

void bench_Dot(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d, DataType * result_d
    );

void bench_WAXPBY(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d, DataType *w,
    DataType alpha, DataType beta
    );


void bench_SymGS(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );

void bench_SymGS(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    );

    
#endif // BENCHMARK_HPP