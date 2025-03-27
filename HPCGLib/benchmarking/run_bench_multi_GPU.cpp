#include "benchmark.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

#include "HPCG_versions/non_blocking_mpi_halo_exchange.cuh"
#include "benchmark_multi_GPU.cpp"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"

//number of processes in x, y, z
#define NPX 2
#define NPY 2
#define NPZ 2
//each process gets assigned problem size of NX x NY x NZ
#define NX 3
#define NY 3
#define NZ 3

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

    // generate a timestamped folder
    std::string base_path = "../../../timing_results/";
    base_path = "../../../dummy_timing_results/";

    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    //start timer
    std::cout << "Starting Benchmark" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    non_blocking_mpi_Implementation<DataType> MGPU_implementation;
    Problem problem = MGPU_implementation.init_comm(argc, argv, NPX, NPY, NPZ, NX, NY, NZ);
    
    //set Device
    InitGPU(&problem);
    
    //start timer
    std::cout << "Starting multi GPU Benchmarks" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    non_blocking_mpi_Implementation<double> MGPU_implementation;

    //run
    run_multi_GPU_benchmarks(2, 2, 1, 16, 16, 16, folder_path, MGPU_implementation, &problem);
    
    //end timer
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end-start;
    minutes = static_cast<int>(elapsed_seconds.count()) / 60;
    seconds = static_cast<int>(elapsed_seconds.count()) % 60;
    std::cout << "Multi GPU Benchmarks took: " << minutes << "m" << seconds << "s" << std::endl;

    MPGU_implementation.finalize_comm(&problem);

    //end timer
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = total_end - total_start;
    minutes = static_cast<int>(duration.count()) / 60;
    seconds = static_cast<int>(duration.count()) % 60;
    std::cout << "Total benchmark duration: " << minutes << " minutes and " << seconds << " seconds." << std::endl;

    std::cout << "Finished Benchmark" << std::endl;


    return 0;
}