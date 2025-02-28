#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "TimingLib/timer.hpp"

CudaTimer::CudaTimer(
        int nx,
        int ny,
        int nz,
        int nnz,
        std::string ault_node,
        std::string matrix_type,
        std::string version_name,
        std::string additional_parameters,
        // this folderpath leads to a timestamped folder, this way we only get a single folder per benchrun
        std::string folder_path
    ) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        this->nx = nx;
        this->ny = ny;
        this->nz = nz;
        this->nnz = nnz;
        this->ault_node = ault_node;
        this->matrix_type = matrix_type;
        this->version_name = version_name;
        this->additional_parameters = additional_parameters;
        this->folder_path = folder_path;
 }

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    writeResultsToCsv();
}

void CudaTimer::startTimer() {
    cudaEventRecord(start, 0);
}

void CudaTimer::stopTimer(std::string method_name) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (method_name == "compute_CG_noPreconditioning") CG_noPreconditioning_times.push_back(milliseconds);
    else if (method_name == "compute_CG") CG_times.push_back(milliseconds);
    else if (method_name == "compute_MG") MG_times.push_back(milliseconds);
    else if (method_name == "compute_SymGS") SymGS_times.push_back(milliseconds);
    else if (method_name == "compute_SPMV") SPMV_times.push_back(milliseconds);
    else if (method_name == "compute_Dot") Dot_times.push_back(milliseconds);
    else if (method_name == "compute_WAXPBY") WAXPBY_times.push_back(milliseconds);
    else{
        std::cerr << "Invalid method name " << method_name << std::endl;
        return;
    }
}

float CudaTimer::getElapsedTime() const {
    return milliseconds;
}

void CudaTimer::writeCSV(std::string filepath, std::string file_header, std::vector<float> times){
    

    // if the vector is empty we don't need to write anything
    if (times.empty()) return;

    // Open the CSV file in append mode
    std::ofstream csv_file(filepath, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filepath << std::endl;
        return;
    }

    // Check if the file is empty and write the header if it is
    std::ifstream infile(filepath);
    infile.seekg(0, std::ios::end);
    if (infile.tellg() == 0) {
        csv_file << file_header << std::endl;
    }
    infile.close();

    // Write the timing results to the CSV file
    for (const auto& time : times) {
        csv_file << time << std::endl;
    }

    // Close the CSV file
    csv_file.close();
}

void CudaTimer::writeResultsToCsv() {

    // std::cout << "Writing results to CSV all of them" << std::endl;

    std::string dim_sting = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);

    base_filename = folder_path
                + version_name + "_"
                + ault_node + "_"
                + matrix_type + "_"
                + ault_node + "_"
                + dim_sting + "_";

    base_fileheader = version_name + ","
                + ault_node + ","
                + matrix_type + ","
                + std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz) + "," + std::to_string(nnz) + ",";

    writeCSV(base_filename + "CG.csv", base_fileheader + "CG," + additional_parameters, CG_times);
    writeCSV(base_filename + "CG_noPreconditioning.csv", base_fileheader + "CG_noPreconditioning," + additional_parameters, CG_noPreconditioning_times);
    writeCSV(base_filename + "MG.csv", base_fileheader + "MG," + additional_parameters, MG_times);
    writeCSV(base_filename + "SymGS.csv", base_fileheader + "SymGS," + additional_parameters, SymGS_times);
    writeCSV(base_filename + "SPMV.csv", base_fileheader + "SPMV," + additional_parameters, SPMV_times);
    writeCSV(base_filename + "Dot.csv", base_fileheader + "Dot," + additional_parameters, Dot_times);
    writeCSV(base_filename + "WAXPBY.csv", base_fileheader + "WAXPBY," + additional_parameters, WAXPBY_times);
    
}