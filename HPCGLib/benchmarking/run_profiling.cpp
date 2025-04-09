#include "benchmark.hpp"
// #include "HPCGLib.hpp"
#include <cuda_profiler_api.h>

#include <iostream>
#include <unordered_map>
#include <stdexcept> // For std::invalid_argument
#include <string>


HPCG_functions<DataType>& get_implementation(const std::string& implementation_name) {
    if (implementation_name == "striped_warp_reduction_Implementation") {
        static striped_warp_reduction_Implementation<DataType> SWR_implementation;
        return SWR_implementation;
    } else if (implementation_name == "striped_coloring_Implementation") {
        static striped_coloring_Implementation<DataType> SC_implementation;
        return SC_implementation;
    } else if (implementation_name == "striped_box_coloring_Implementation") {
        static striped_box_coloring_Implementation<DataType> SBC_implementation;
        return SBC_implementation;
    } else if (implementation_name == "striped_coloring_precomputed_Implementation") {
        static striped_coloringPrecomputed_Implementation<DataType> SCP_implementation;
        return SCP_implementation;
    } else if (implementation_name == "no_store_striped_coloring_Implementation") {
        static no_store_striped_coloring_Implementation<DataType> NC_SC_implementation;
        return NC_SC_implementation;
    } else if (implementation_name == "striped_COR_box_coloring_Implementation") {
        static striped_COR_box_coloring_Implementation<DataType> SBC_COR_implementation;
        return SBC_COR_implementation;
    } else {
        throw std::invalid_argument("Unknown implementation name: " + implementation_name);
    }
}

int main(int argc, char* argv[]){

    // int device;
    // cudaGetDevice(&device);

    // int l2CacheSize;
    // cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, device);

    // std::cout << "L2 Cache Size: " << static_cast<double>(l2CacheSize) / (1024*1024) << " MB" << std::endl;

    // Check if the correct number of arguments is provided
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <nx> <ny> <nz>" << std::endl;
        return 1;
    }
    std::cout << "Starting profiling" << std::endl;

    int nx = std::stoi(argv[1]);
    int ny = std::stoi(argv[2]);
    int nz = std::stoi(argv[3]);

    std::string function_name = argv[4];
    std::string implementation_name = argv[5];

    HPCG_functions<DataType>& implementation = get_implementation(implementation_name);

    implementation.set_maxCGIters(1);
    implementation.set_maxSymGSIters(1);

    striped_Matrix<DataType> striped_A;
    striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);
    
    local_int_t num_rows = striped_A.get_num_rows();
    local_int_t num_cols = striped_A.get_num_cols();
    local_int_t nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();
    
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<DataType> x (num_rows, 0.0);
    std::vector<DataType> a = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(num_rows, RANDOM_SEED);

    if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
        // initialize the MG data
        striped_Matrix <DataType>* current_matrix = &striped_A;
        for(int i = 0; i < 3; i++){
            current_matrix->initialize_coarse_Matrix();
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    }

    // Allocate the memory on the device
    DataType * a_d;
    DataType * b_d;
    DataType * x_d;
    DataType * y_d;
    DataType * result_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    // now comes the actual profiling
    if(implementation.CG_implemented && function_name == "CG"){
        
        int n_iters = 0;
        double normr;
        double normr0;

        // implementation.bx = 3;
        // implementation.by = 3;
        // implementation.bz = 3;
        
        // start the profiling
        cudaProfilerStart();
        implementation.compute_CG(
            striped_A,
            y_d, x_d,
            n_iters, normr, normr0
        );
        cudaProfilerStop();
    }

    if (implementation.SymGS_implemented && function_name == "SymGS") {
        std::cout << "SymGS" << std::endl;
        // start the profiling
        cudaProfilerStart();
        implementation.compute_SymGS(
            striped_A,
            y_d, x_d
        );
        cudaProfilerStop();
    }

    if (implementation.SPMV_implemented && function_name == "SPMV") {
        // start the profiling
        cudaProfilerStart();
        implementation.compute_SPMV(
            striped_A,
            a_d, x_d
        );
        cudaProfilerStop();
    }

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);
    

}