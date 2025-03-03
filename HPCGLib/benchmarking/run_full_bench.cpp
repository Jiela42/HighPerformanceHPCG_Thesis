#include "benchmark.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>


namespace fs = std::filesystem;

std::string createTimestampedFolder(const std::string base_folder){
    if(!fs::exists(base_folder)){
        std::cout << "Base folder " << base_folder <<" does not exist" << std::endl;
    }
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");

    std::string folder_path = base_folder + ss.str();
    fs::create_directory(folder_path);

    return folder_path;

}


int main() {

    // make a CSR matrix on the gpu
    // sparse_CSR_Matrix<double> A;
    // // A.generateMatrix_onGPU(128, 128, 128);
    // std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(128, 128, 128);
    // A = problem.first;

    // std::cout << "nnz: " << A.get_nnz() << std::endl;

    // // lets copy this by hand
    // int * row_ptr_d;
    // int * col_idx_d;
    // double * values_d;

    // CHECK_CUDA(cudaMalloc(&row_ptr_d, (A.get_num_rows() + 1) * sizeof(int)));
    // CHECK_CUDA(cudaMalloc(&col_idx_d, A.get_nnz() * sizeof(int)));
    // CHECK_CUDA(cudaMalloc(&values_d, A.get_nnz() * sizeof(double)));

    // A.copy_Matrix_toGPU();


    // sparse_CSR_Matrix<double> B;

    // B.generateMatrix_onGPU(128, 128, 128);

    // striped_Matrix<double> striped_A;
    // striped_A.striped_Matrix_from_sparse_CSR(A);

    // generate a timestamped folder
    std::string base_path = "../../../timing_results/";
    base_path = "../../../dummy_timing_results/";

    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    std::cout << "Starting Benchmark" << std::endl;

    std::cout << "Starting cuSparse 3d27p Benchmarks" << std::endl;
    // cuSparse_Implementation<double> CSR_implementation;
    // run_cuSparse_3d27p_benchmarks(8, 8, 8, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(16, 16, 16, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(32, 32, 32, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(64, 64, 64, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(128, 128, 64, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(128, 64, 64, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(128, 128, 128, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(256, 128, 128, folder_path, CSR_Implementation);
    // run_cuSparse_3d27p_benchmarks(256, 256, 128, folder_path, CSR_Implementation);
    // std::cout << "Finished cuSparse 3d27p Benchmarks 256 256 128" << std::endl;
    // run_cuSparse_3d27p_benchmarks(256, 256, 256, folder_path, CSR_Implementation);

    std::cout << "Starting naive Striped 3d27p Benchmarks" << std::endl;
    // naiveStriped_Implementation<double> NS_implementation;
    // run_naiveStriped_3d27p_benchmarks(8, 8, 8, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(16, 16, 16, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(32, 32, 32, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(64, 64, 64, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(128, 64, 64, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(128, 128, 64, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(128, 128, 128, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(256, 128, 128, folder_path, NS_implementation);
    // run_naiveStriped_3d27p_benchmarks(256, 256, 128, folder_path, NS_implementation);

    std::cout << "Starting Striped Shared Memory 3d27p Benchmarks" << std::endl;
    // Striped_Shared_Memory_Implementation<double> SSM_implementation;
    // run_stripedSharedMem_3d27p_benchmarks(8, 8, 8, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(16, 16, 16, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(32, 32, 32, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(64, 64, 64, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(128, 64, 64, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(128, 128, 64, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(128, 128, 128, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(256, 128, 128, folder_path, SSM_implementation);
    // run_stripedSharedMem_3d27p_benchmarks(256, 256, 128, folder_path, SSM_implementation);

    std::cout << "Starting Striped Warp Reduction 3d27p Benchmarks" << std::endl;
    striped_warp_reduction_Implementation<double> SWR_implementation;
    run_striped_warp_reduction_3d27p_benchmarks(8, 8, 8, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(16, 16, 16, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(32, 32, 32, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(64, 64, 64, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(128, 64, 64, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(128, 128, 64, folder_path, SWR_implementation);
    std::cout << "let's do the big one now" << std::endl;
    run_striped_warp_reduction_3d27p_benchmarks(128, 128, 128, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(256, 128, 128, folder_path, SWR_implementation);
    // run_striped_warp_reduction_3d27p_benchmarks(256, 256, 128, folder_path, SWR_implementation);


    // this version has issues, so we never run it and we also don't plan on fixing it because we are benches
    // std::cout << "Starting Striped Preprocessed 3d27p Benchmarks" << std::endl;
    // striped_preprocessed_Implementation<double> SPP_implementation;
    // run_striped_preprocessed_3d27p_benchmarks(8, 8, 8, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_benchmarks(16, 16, 16, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_benchmarks(32, 32, 32, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_benchmarks(64, 64, 64, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_benchmarks(128, 64, 64, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_benchmarks(128, 128, 64, folder_path, SPP_implementation);
    // run_striped_preprocessed_3d27p_benchmarks(128, 128, 128, folder_path, SPP_implementation);

    std::cout << "Starting Striped Colored 3d27p Benchmarks" << std::endl;
    striped_coloring_Implementation<double> SC_implementation;
    // run_striped_coloring_3d27p_benchmarks(8, 8, 8, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(16, 16, 16, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(24, 24, 24, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(32, 32, 32, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(64, 64, 64, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(128, 64, 64, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(128, 128, 64, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(128, 128, 128, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(256, 128, 128, folder_path, SC_implementation);
    // run_striped_coloring_3d27p_benchmarks(256, 256, 128, folder_path, SC_implementation);

    std::cout << "Starting no store striped coloring 3d27p Benchmarks" << std::endl;
    no_store_striped_coloring_Implementation<double> NC_SC_implementation;

    // run_no_store_striped_coloring_3d27p_benchmarks(8, 8, 8, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(16, 16, 16, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(24, 24, 24, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(32, 32, 32, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(64, 64, 64, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(128, 64, 64, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(128, 128, 64, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(128, 128, 128, folder_path, NC_SC_implementation);
    // run_no_store_striped_coloring_3d27p_benchmarks(256, 128, 128, folder_path, NC_SC_implementation);
    
    std::cout << "Starting striped coloring precomputed 3d27p Benchmarks" << std::endl;
    striped_coloringPrecomputed_Implementation<double> SCP_implementation;
    run_striped_coloringPrecomputed_3d27p_benchmarks(8, 8, 8, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(16, 16, 16, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(32, 32, 32, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(64, 64, 64, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(128, 64, 64, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(128, 128, 64, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(128, 128, 128, folder_path, SCP_implementation);
    // run_striped_coloringPrecomputed_3d27p_benchmarks(256, 128, 128, folder_path, SCP_implementation);

    std::cout << "Starting striped box coloring 3d27p Benchmarks" << std::endl;
    striped_box_coloring_Implementation<double> SBC_implementation;
    // run_striped_box_coloring_3d27p_benchmarks(8, 8, 8, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(16, 16, 16, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(24, 24, 24, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(32, 32, 32, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(64, 64, 64, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(128, 64, 64, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(128, 128, 64, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(128, 128, 128, folder_path, SBC_implementation);
    // run_striped_box_coloring_3d27p_benchmarks(256, 128, 128, folder_path, SBC_implementation);
    

    std::cout << "Finished Benchmark" << std::endl;

    return 0;
}