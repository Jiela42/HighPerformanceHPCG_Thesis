#ifndef TIMER_HPP
#define TIMER_HPP

#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "UtilLib/utils.hpp"

class Timer {
public:
    // Constructors
    Timer();
    Timer(
        int nx,
        int ny,
        int nz,
        local_int_t nnz,
        std::string ault_node,
        std::string matrix_type,
        std::string version_name,
        std::string additional_parameters,
        std::string folder_path
    );

    // Destructor
    ~Timer();

    // Function to start the timer
    virtual void startTimer() = 0;

    // Function to stop the timer and record the elapsed time
    virtual void stopTimer(std::string method_name) = 0;

    void add_additional_parameters(std::string another_parameter);


protected:
    // Function to write the timing results to a CSV file
    void writeResultsToCsv();
    void writeCSV(std::string filepath, std::string file_header, std::vector<float> times);
    // virtual float getElapsedTime() const = 0;

    // float milliseconds;
    // the filename also includes the method name ;)
    std::string base_filename;
    std::string base_fileheader;
    std::string folder_path;
    int nx, ny, nz;
    local_int_t nnz;
    std::string ault_node;
    std::string matrix_type;
    std::string version_name;
    std::string additional_parameters;

    std::vector<float> CG_times;
    std::vector<float> CG_noPreconditioning_times;
    std::vector<float> MG_times;
    std::vector<float> SymGS_times;
    std::vector<float> SPMV_times;
    std::vector<float> Dot_times;
    std::vector<float> WAXPBY_times;
    std::vector<float> ExchangeHalo_times;
};

#endif // TIMER_HPP