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

    run_cuSparse_3d27p_SymGS_benchmark(64, 64, 64, folder_path);

    // run_warp_reduction_3d27p_Dot_benchmark(64, 64, 64, folder_path);
    // run_warp_reduction_3d27p_SPMV_benchmark(64, 64, 64, folder_path);



    std::cout << "Finished Benchmark" << std::endl;  

}