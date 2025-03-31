#ifndef CUDATIMER_HPP
#define CUDATIMER_HPP

#include "timer.hpp"
#include <cuda_runtime.h>

class CudaTimer : public Timer {
public:
    // Constructor
    CudaTimer(
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
    ~CudaTimer();

    // Function to start the timer
    void startTimer() override;

    // Function to stop the timer and record the elapsed time
    void stopTimer(std::string method_name) override;
    
private:
    cudaEvent_t start, stop;
    // float getElapsedTime() const override;

};

#endif // CUDATIMER_HPP
