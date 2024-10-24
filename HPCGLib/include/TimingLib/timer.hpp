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

    // Function to write the timing results to a CSV file
    void writeResultsToCsv(const std::string& csv_file, const std::string& method_name, int n, int m, int num_iterations, const std::vector<float>& results);

private:
    // CUDA events for timing
    cudaEvent_t start;
    cudaEvent_t stop;

    // Member variables for storing metadata
    int nx;
    int ny;
    int nz;
    int nnz;
    std::string ault_node;
    std::string matrix_type;
    std::string version_name;
    std::string folder_path;

    // Vector to store timing results
    std::vector<float> times;
};

#endif // TIMER_HPP