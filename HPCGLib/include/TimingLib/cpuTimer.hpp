#ifndef CPUTIMER_HPP
#define CPUTIMER_HPP

#include "timer.hpp"
#include <cuda_runtime.h>

class cpuTimer : public Timer {
public:
    // Constructor
    cpuTimer(
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
    ~cpuTimer();

    // Function to start the timer
    void startTimer() override;

    // Function to stop the timer and record the elapsed time
    void stopTimer(std::string method_name) override;
    
private:
    float t_start;

};

#endif // CPUTIMER_HPP
