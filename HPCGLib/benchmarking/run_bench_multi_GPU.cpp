#include "benchmark.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

#include "HPCG_versions/non_blocking_mpi_halo_exchange.cuh"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"

using DataType = double;

namespace fs = std::filesystem;

std::string createTimestampedFolder(const std::string base_folder){
    if(!fs::exists(base_folder)){
        std::cout << "Base folder " << base_folder <<" does not exist" << std::endl;
    }
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");

    std::string folder_path = base_folder + ss.str();
    fs::create_directory(folder_path);

    return folder_path;

}


int main(int argc, char *argv[]) {

    int NPX = 2;
    int NPY = 2;
    int NPZ = 2;
    int NX = 16;
    int NY = 16;
    int NZ = 16;

    //read input from console
    std::vector<int*> argVars = {&NPX, &NPY, &NPZ, &NX, &NY, &NZ};
    std::vector<std::string> argNames = {"NPX", "NPY", "NPZ", "NX", "NY", "NZ"};
    for (size_t i = 0; i < argVars.size(); i++) {
        if (argc > static_cast<int>(i + 1)) {
            try {
                *argVars[i] = std::stoi(argv[i + 1]);
            } catch (...) {
                std::cerr << "Invalid " << argNames[i] << " argument, using default value " << *argVars[i] << std::endl;
            }
        }
    }

    // Now, process additional arguments for benchmark selection.
    // Default is "ALL". If additional arguments exist, join them into a string.
    std::string benchFilter = "ALL";
    if (argc > 7) {
        std::ostringstream oss;
        for (int i = 7; i < argc; i++) {
            if (i > 7) oss << " ";
            oss << argv[i];
        }
        benchFilter = oss.str();
    }

    // generate a timestamped folder
    std::string base_path = "../../../timing_results/";
    base_path = "../../../dummy_timing_results/";

    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    //start timer
    auto total_start = std::chrono::high_resolution_clock::now();
    
    non_blocking_mpi_Implementation<DataType> MGPU_Implementation;
    Problem *problem = MGPU_Implementation.init_comm(argc, argv, NPX, NPY, NPZ, NX, NY, NZ);
    if(problem->rank == 0) std::cout << "Starting Benchmark" << std::endl;
    
    //start timer
    if(problem->rank == 0) std::cout << "Starting multi GPU Benchmarks" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    non_blocking_mpi_Implementation<double> MGPU_implementation;

    //run
    run_multi_GPU_benchmarks(NPX, NPY, NPZ, NX, NY, NZ, folder_path, MGPU_Implementation, problem, benchFilter);
    
    //end timer
    auto end = std::chrono::high_resolution_clock::now();
    // Convert to seconds;
    std::chrono::duration<double> elapsed_seconds = end - start;
    int minutes = static_cast<int>(elapsed_seconds.count()) / 60;
    int seconds = static_cast<int>(elapsed_seconds.count()) % 60;
    if(problem->rank == 0) std::cout << "Multi GPU Benchmarks took: " << minutes << "m" << seconds << "s" << std::endl;

    MGPU_Implementation.finalize_comm(problem);

    //end timer
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = total_end - total_start;
    minutes = static_cast<int>(duration.count()) / 60;
    seconds = static_cast<int>(duration.count()) % 60;
    if(problem->rank == 0) std::cout << "Total benchmark duration: " << minutes << " minutes and " << seconds << " seconds." << std::endl;

    if(problem->rank == 0) std::cout << "Finished Benchmark" << std::endl;


    return 0;
}