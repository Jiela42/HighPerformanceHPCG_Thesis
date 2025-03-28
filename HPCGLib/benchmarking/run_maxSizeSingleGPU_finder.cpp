
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


int main(){
    std::string base_path = "../../../dummy_timing_results/";

    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    striped_coloring_Implementation<double> implementation;

    for(int i = 7; i < 50; i++){
        int nx = i*8;
        int ny = i*8;
        int nz = i*8;

        std::cout << "Running striped coloring benchmarks for " << nx << " " << ny << " " << nz << std::endl;
        run_striped_coloring_3d27p_benchmarks(nx, ny, nz, folder_path, implementation);
        std::cout << "Finished striped coloring benchmarks for " << nx << " " << ny << " " << nz << std::endl;
    }



    std::cout << "Starting Finder" << std::endl;



}