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
    base_path = "../../../dummy_timing_results/";


    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    std::cout << "Starting Benchmark" << std::endl;

    run_cuSparse_3d27p_SymGS_benchmark(8, 8, 8, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(16, 16, 16, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(32, 32, 32, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(64, 64, 64, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(128, 64, 64, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(128, 128, 64, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(128, 128, 128, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(256, 128, 128, folder_path);
    // run_cuSparse_3d27p_SymGS_benchmark(256, 256, 128, folder_path);


    // std::cout << "Finished cuSparse Benchmark" << std::endl;
    run_warp_reduction_3d27p_Dot_benchmark(8, 8, 8, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(16, 16, 16, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(24, 24, 24, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(32, 32, 32, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(64, 64, 64, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(128, 64, 64, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(128, 128, 64, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(128, 128, 128, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(256, 128, 128, folder_path);
    // run_warp_reduction_3d27p_Dot_benchmark(256, 256, 128, folder_path);
    run_warp_reduction_3d27p_SPMV_benchmark(8, 8, 8, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(16, 16, 16, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(24, 24, 24, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(32, 32, 32, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(64, 64, 64, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(128, 64, 64, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(128, 128, 64, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(128, 128, 128, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(256, 128, 128, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(256, 256, 128, folder_path);

    run_warp_reduction_3d27p_SymGS_benchmark(8, 8, 8, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(16, 16, 16, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(32, 32, 32, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(64, 64, 64, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(128, 64, 64, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(128, 128, 64, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(128, 128, 128, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(256, 128, 128, folder_path);
    // run_warp_reduction_3d27p_SymGS_benchmark(256, 256, 128, folder_path);

    run_striped_preprocessed_3d27p_SymGS_benchmark(8, 8, 8, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(16, 16, 16, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(32, 32, 32, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(64, 64, 64, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(128, 64, 64, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(128, 128, 64, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(128, 128, 128, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(256, 128, 128, folder_path);
    // run_striped_preprocessed_3d27p_SymGS_benchmark(256, 256, 128, folder_path);

    


    std::cout << "Finished Benchmark" << std::endl;  

}