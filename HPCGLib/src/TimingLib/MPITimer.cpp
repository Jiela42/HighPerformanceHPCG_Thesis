#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "mpi.h"
#include "time.h"

// #include "TimingLib/timer.hpp"
#include "TimingLib/MPITimer.hpp"
#include "UtilLib/cuda_utils.hpp"

MPITimer::MPITimer(
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
        this->t_start = 0.0;
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

 void reduceVector(std::vector<float>& local_times){
     int n = local_times.size();
     for(int i = 0; i < n; i++){
         float local_time = local_times[i];
         float global_time;
         MPI_Reduce(&local_time, &global_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
         local_times[i] = global_time;
     }
}

MPITimer::~MPITimer() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //find the per measurement
    reduceVector(CG_times);
    reduceVector(CG_noPreconditioning_times);
    reduceVector(MG_times);
    reduceVector(SymGS_times);
    reduceVector(SPMV_times);
    reduceVector(Dot_times);
    reduceVector(WAXPBY_times);
    reduceVector(ExchangeHalo_times);

    //only one csv file is written
    if (rank == 0) {
        writeResultsToCsv();
    }
}


void MPITimer::startTimer() {
    CHECK_CUDA(cudaDeviceSynchronize());
    this->t_start = MPI_Wtime();
}

void MPITimer::stopTimer(std::string method_name) {
    CHECK_CUDA(cudaDeviceSynchronize());
    float t_end = MPI_Wtime();

    float milliseconds = (t_end - this->t_start) * 1000.0;


    if (method_name == "compute_CG_noPreconditioning") CG_noPreconditioning_times.push_back(milliseconds);
    else if (method_name == "compute_CG") CG_times.push_back(milliseconds);
    else if (method_name == "compute_MG") MG_times.push_back(milliseconds);
    else if (method_name == "compute_SymGS") SymGS_times.push_back(milliseconds);
    else if (method_name == "compute_SPMV") SPMV_times.push_back(milliseconds);
    else if (method_name == "compute_Dot") Dot_times.push_back(milliseconds);
    else if (method_name == "compute_WAXPBY") WAXPBY_times.push_back(milliseconds);
    else if (method_name == "ExchangeHalo") ExchangeHalo_times.push_back(milliseconds);
    else{
        std::cerr << "Invalid method name " << method_name << std::endl;
        return;
    }
}