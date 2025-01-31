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

    // std::cout << "Starting cuSparse 3d27p Benchmarks" << std::endl;
    // run_cuSparse_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_cuSparse_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_cuSparse_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_cuSparse_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_cuSparse_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_cuSparse_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_cuSparse_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_cuSparse_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_cuSparse_3d27p_benchmarks(256, 256, 128, folder_path);
    // std::cout << "Finished cuSparse 3d27p Benchmarks 256 256 128" << std::endl;
    // run_cuSparse_3d27p_benchmarks(256, 256, 256, folder_path);

    // std::cout << "Starting naive Striped 3d27p Benchmarks" << std::endl;
    // run_naiveStriped_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_naiveStriped_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_naiveStriped_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_naiveStriped_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_naiveStriped_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_naiveStriped_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_naiveStriped_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_naiveStriped_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_naiveStriped_3d27p_benchmarks(256, 256, 128, folder_path);

    // std::cout << "Starting Striped Shared Memory 3d27p Benchmarks" << std::endl;
    // run_stripedSharedMem_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_stripedSharedMem_3d27p_benchmarks(256, 256, 128, folder_path);

    // std::cout << "Starting Striped Warp Reduction 3d27p Benchmarks" << std::endl;
    // run_striped_warp_reduction_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_striped_warp_reduction_3d27p_benchmarks(256, 256, 128, folder_path);

    // std::cout << "Starting Striped Preprocessed 3d27p Benchmarks" << std::endl;
    // run_striped_preprocessed_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_striped_preprocessed_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_striped_preprocessed_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_striped_preprocessed_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_striped_preprocessed_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_striped_preprocessed_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_striped_preprocessed_3d27p_benchmarks(128, 128, 128, folder_path);

    // std::cout << "Starting Striped Colored 3d27p Benchmarks" << std::endl;
    // run_striped_coloring_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_striped_coloring_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_striped_coloring_3d27p_benchmarks(24, 24, 24, folder_path);
    // run_striped_coloring_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_striped_coloring_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_striped_coloring_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_striped_coloring_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_striped_coloring_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_striped_coloring_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_striped_coloring_3d27p_benchmarks(256, 256, 128, folder_path);

    // std::cout << "Starting no store striped coloring 3d27p Benchmarks" << std::endl;
    // run_no_store_striped_coloring_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(24, 24, 24, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_no_store_striped_coloring_3d27p_benchmarks(256, 256, 128, folder_path);
    
    // std::cout << "Starting striped coloring precomputed 3d27p Benchmarks" << std::endl;
    // run_striped_coloringPrecomputed_3d27p_benchmarks(8, 8, 8, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(16, 16, 16, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(32, 32, 32, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(64, 64, 64, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(128, 64, 64, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(128, 128, 64, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(128, 128, 128, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(256, 256, 128, folder_path);

    std::cout << "Starting striped box coloring 3d27p Benchmarks" << std::endl;
    run_striped_box_coloring_3d27p_benchmarks(8, 8, 8, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(16, 16, 16, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(24, 24, 24, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(32, 32, 32, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(64, 64, 64, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(128, 64, 64, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(128, 128, 64, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(128, 128, 128, folder_path);
    run_striped_box_coloring_3d27p_benchmarks(256, 128, 128, folder_path);
    // run_striped_box_coloring_3d27p_benchmarks(256, 256, 128, folder_path);
    

    std::cout << "Finished Benchmark" << std::endl;

    return 0;
}