#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

class CudaTimer {
public:
    CudaTimer(
        int nx,
        int ny,
        int nz,
        int nnz,
        std::string ault_node,
        std::string matrix_type,
        std::string version_name,
        // this folderpath leads to a timestamped folder, this way we only get a single folder per benchrun
        std::string folder_path
    ) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        writeResultsToCsv();
    }

    void startTimer() {
        cudaEventRecord(start, 0);
    }

    void stopTimer(std::string method_name) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (method_name == "compute_CG") CG_times.push_back(milliseconds);
        else if (method_name == "compute_MG") MG_times.push_back(milliseconds);
        else if (method_name == "compute_SymGS") SymGS_times.push_back(milliseconds);
        else if (method_name == "compute_SPMV") SPMV_times.push_back(milliseconds);
        else if (method_name == "compute_Dot") Dot_times.push_back(milliseconds);
        else{
            std::cerr << "Invalid method name" << method_name << std::endl;
            return;
        }
    }

    float getElapsedTime() const {
        return milliseconds;
    }

private:

    void writeCSV(std::string filepath, std::string file_header, std::vector<float> times){
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
    
    void writeResultsToCsv() {

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
                    + ault_node + ","
                    + std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz) + "," + std::to_string(nnz) + ",";

        writeCSV(base_filename + "CG.csv", base_fileheader + "CG,", CG_times);
        writeCSV(base_filename + "MG.csv", base_fileheader + "MG,", MG_times);
        writeCSV(base_filename + "SymGS.csv", base_fileheader + "SymGS,", SymGS_times);
        writeCSV(base_filename + "SPMV.csv", base_fileheader + "SPMV,", SPMV_times);
        writeCSV(base_filename + "Dot.csv", base_fileheader + "Dot,", Dot_times);
        
    }

    cudaEvent_t start, stop;
    float milliseconds = 0;
    // the filename also includes the method name ;)
    std::string base_filename;
    std::string base_fileheader;
    std::string folder_path;
    int nx, ny, nz, nnz;
    std::string ault_node;
    std::string matrix_type;
    std::string version_name;

    std::vector<float> CG_times;
    std::vector<float> MG_times;
    std::vector<float> SymGS_times;
    std::vector<float> SPMV_times;
    std::vector<float> Dot_times;

};