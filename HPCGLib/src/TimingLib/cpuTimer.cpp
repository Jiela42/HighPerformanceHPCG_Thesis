#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

// #include "TimingLib/timer.hpp"
#include "TimingLib/cudaTimer.hpp"

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

    writeResultsToCsv();
}

void CudaTimer::startTimer() {
    // @Nils
    // Please implement for CPU
}

void CudaTimer::stopTimer(std::string method_name) {

    float milliseconds = 0.0;

    // @Nils
    // Please implement for CPU
    // write the measured milliseconds into the milliseconds variable


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