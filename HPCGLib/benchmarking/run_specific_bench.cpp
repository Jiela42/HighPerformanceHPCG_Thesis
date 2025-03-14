#include "benchmark.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>


namespace fs = std::filesystem;

std::string createTimestampedFolder(const std::string base_folder){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");

    std::string folder_path = base_folder + ss.str();
    fs::create_directory(folder_path);

    return folder_path;

}


int main() {

    // generate a timestamped folder
    std::string base_path = "../../../timing_results/";
    // base_path = "../../../dummy_timing_results/";


    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    std::cout << "Starting Benchmark" << std::endl;

    cuSparse_Implementation<double> CSR_implementation;
    // run_cuSparse_3d27p_SPMV_benchmark(64, 64, 64, folder_path, CSR_implementation);


    // run_cuSparse_3d27p_SymGS_benchmark(8, 8, 8, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(16, 16, 16, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(32, 32, 32, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(64, 64, 64, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(128, 64, 64, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(128, 128, 64, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(128, 128, 128, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(256, 128, 128, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_SymGS_benchmark(256, 256, 128, folder_path, CSR_implementation);

    // run_cuSparse_3d27p_Dot_benchmark(8, 8, 8, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(16, 16, 16, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(24, 24, 24, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(32, 32, 32, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(64, 64, 64, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(128, 64, 64, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(128, 128, 64, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(128, 128, 128, folder_path, CSR_implementation);
    // run_cuSparse_3d27p_Dot_benchmark(256, 128, 128, folder_path, CSR_implementation);
    std::cout << "Finished cuSparse Benchmark" << std::endl;

    striped_warp_reduction_Implementation<double> SWR_implementation;
    // run_warp_reduction_3d27p_Dot_benchmark(8, 8, 8, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(16, 16, 16, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(24, 24, 24, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(32, 32, 32, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(64, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(128, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(128, 128, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(128, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(256, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_Dot_benchmark(256, 256, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(8, 8, 8, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(16, 16, 16, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(24, 24, 24, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(32, 32, 32, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(64, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(128, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(128, 128, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(128, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(256, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(256, 256, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_WAXPBY_benchmark(256, 256, 256, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(8, 8, 8, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(16, 16, 16, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(24, 24, 24, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(32, 32, 32, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(64, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(128, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(128, 128, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(128, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(256, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SPMV_benchmark(256, 256, 128, folder_path, SWR_implementation);

    // run_warp_reduction_3d27p_SymGS_benchmark(8, 8, 8, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(16, 16, 16, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(32, 32, 32, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(64, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(128, 64, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(128, 128, 64, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(128, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(256, 128, 128, folder_path, SWR_implementation);
    // run_warp_reduction_3d27p_SymGS_benchmark(256, 256, 128, folder_path, SWR_implementation);

    // this version has issues, so we never run it and we also don't plan on fixing it because we are benches
    // std::cout << "Starting Striped Preprocessed 3d27p Benchmarks" << std::endl;
    // striped_preprocessed_Implementation<double> SPP_implementation;
    // run_striped_preprocessed_3d27p_SymGS_benchmark(8, 8, 8, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(16, 16, 16, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(32, 32, 32, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(64, 64, 64, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(128, 64, 64, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(128, 128, 64, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(128, 128, 128, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(256, 128, 128, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(256, 256, 128, folder_path, SPP_implementation);

    striped_coloring_Implementation<double> SC_implementation;
    // run_striped_coloring_3d27p_SymGS_benchmark(8, 8, 8, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_SymGS_benchmark(64, 64, 64, folder_path, SC_implementation);

    no_store_striped_coloring_Implementation<double> NC_SC_implementation;
    // run_no_store_striped_coloring_3d27p_SymGS_benchmark(8, 8, 8, folder_path, NC_SC_implementation);

    striped_coloringPrecomputed_Implementation<double> SCP_implementation;
    // run_striped_coloringPrecomputed_3d27p_SymGS_benchmark(8, 8, 8, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_SymGS_benchmark(256, 256, 128, folder_path, SCP_implementation);

    striped_box_coloring_Implementation<double> SBC_implementation;
    run_striped_box_coloring_3d27p_SymGS_benchmark(8, 8, 8, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(16, 16, 16, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(24, 24, 24, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(32, 32, 32, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(64, 64, 64, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(128, 64, 64, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(128, 128, 64, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(128, 128, 128, folder_path, SBC_implementation);
    run_striped_box_coloring_3d27p_SymGS_benchmark(256, 128, 128, folder_path, SBC_implementation);

    // run_striped_box_coloring_3d27p_CG_benchmark(24, 24, 24, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_CG_benchmark(32, 32, 32, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_CG_benchmark(64, 64, 64, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_CG_benchmark(128, 64, 64, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_CG_benchmark(128, 128, 64, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_CG_benchmark(128, 128, 128, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_CG_benchmark(256, 128, 128, folder_path, SBC_implementation);


    std::cout << "Finished Benchmark" << std::endl;  

}