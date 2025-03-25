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
        int nnz,
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

    // @Nils
    // I declared start & stop variables here
    // because they were cuda events and I didn't want to create them all the time
    // If you use floats or another primitive you probably don't need that.


};

#endif // CPUTIMER_HPP
