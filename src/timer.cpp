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
    CudaTimer(const std::string &method_name, int n, int m, int num_iterations)
        : csv_file(generateTimestampedFileName()), method_name(method_name), n(n), m(m), num_iterations(num_iterations) {
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

    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        results.push_back(milliseconds);
    }

    float getElapsedTime() const {
        return milliseconds;
    }

private:
    std::string generateTimestampedFileName() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        return "timing_results_" + ss.str() + ".csv";
    }

    void writeResultsToCsv() {
        std::ofstream csv(csv_file, std::ios::app);
        if (!csv.is_open()) {
            std::cerr << "Failed to open CSV file." << std::endl;
            return;
        }

        // Check if the file is empty and write headers if it is
        std::ifstream infile(csv_file);
        infile.seekg(0, std::ios::end);
        if (infile.tellg() == 0) {
            csv << "Method,Rows,Cols,Iterations,Time(ms)" << std::endl;
        }
        infile.close();

        for (const auto &time : results) {
            csv << method_name << "," << n << "," << m << "," << num_iterations << "," << time << std::endl;
        }
        csv.close();
    }

    cudaEvent_t start, stop;
    float milliseconds = 0;
    std::string csv_file;
    std::string method_name;
    int n, m, num_iterations;
    std::vector<float> results;
};