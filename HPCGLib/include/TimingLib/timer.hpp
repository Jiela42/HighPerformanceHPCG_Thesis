#ifndef TIMER_HPP
#define TIMER_HPP

#include <cuda_runtime.h>
#include <string>
#include <vector>

class CudaTimer {
public:
    // Constructor
    CudaTimer(
        int nx,
        int ny,
        int nz,
        int nnz,
        std::string ault_node,
        std::string matrix_type,
        std::string version_name,
        std::string folder_path
    );

    // Destructor
    ~CudaTimer();

    // Function to start the timer
    void startTimer();

    // Function to stop the timer and record the elapsed time
    void stopTimer(std::string method_name);


private:
    // Function to write the timing results to a CSV file
    void writeResultsToCsv();
    void writeCSV(std::string filepath, std::string file_header, std::vector<float> times);
    float getElapsedTime() const;

    cudaEvent_t start, stop;
    float milliseconds;
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

#endif // TIMER_HPP