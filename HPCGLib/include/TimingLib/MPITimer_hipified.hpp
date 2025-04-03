#ifndef MPITIMER_HPP
#define MPITIMER_HPP

#include "timer_hipified.hpp"
#include <hip/hip_runtime.h>

class MPITimer : public Timer {
public:
    // Constructor
    MPITimer(
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
    ~MPITimer();

    // Function to start the timer
    void startTimer() override;

    // Function to stop the timer and record the elapsed time
    void stopTimer(std::string method_name) override;
    
private:
    float t_start;

};

#endif // MPITIMER_HPP
