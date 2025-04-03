#include "benchmark_hipified.hpp"


void run_striped_COR_box_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_COR_box_coloring_Implementation<DataType>& implementation){
    
    striped_Matrix<DataType> striped_A;
    striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);
    
    local_int_t num_rows = striped_A.get_num_rows();
    local_int_t num_cols = striped_A.get_num_cols();
    local_int_t nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();

    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<DataType> x (num_rows, 0.0);


    striped_A.generate_box_coloring();
    
    // Allocate the memory on the device
    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(hipMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, num_rows * sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(x_d, x.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), num_rows * sizeof(DataType), hipMemcpyHostToDevice));

    for(int i = 3; i <= 3; i ++){
        
        implementation.bx = i;
        implementation.by = i;
        implementation.bz = i;

        std::string box_dims = std::to_string(implementation.bx) + "x" + std::to_string(implementation.by) + "x" + std::to_string(implementation.bz);
        std::string coop_num_string = std::to_string(implementation.SymGS_cooperation_number);
        
        std::string implementation_name = implementation.version_name + " (coloringBox " + box_dims + ")";
        // std::string implementation_name = implementation.version_name;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

        // std::cout << "Running the SymGS benchmark for the implementation: " << implementation_name << std::endl;

        bench_SymGS(implementation, *timer, striped_A, x_d, y_d);
    
        delete timer;
    }

    // free the memory
    CHECK_CUDA(hipFree(x_d));
    CHECK_CUDA(hipFree(y_d));

}

void run_striped_COR_box_coloring_3d27p_CG_benchmark(int nx, int ny, int nz, std::string folder_path, striped_COR_box_coloring_Implementation<DataType>& implementation){
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


    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(hipMalloc(&x_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, num_rows * sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(x_d, x.data(), num_rows * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), num_rows * sizeof(DataType), hipMemcpyHostToDevice));

    striped_A.generate_box_coloring();

    bench_CG(
        implementation,
        *timer,
        striped_A,
        x_d, y_d
    );


    // free da memory
    CHECK_CUDA(hipFree(x_d));
    CHECK_CUDA(hipFree(y_d));

    delete timer;

}

void run_striped_COR_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_COR_box_coloring_Implementation<DataType>& implementation){

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

    striped_A.generate_box_coloring();
    
    std::string box_dims = std::to_string(implementation.bx) + "x" + std::to_string(implementation.by) + "x" + std::to_string(implementation.bz);
    std::string implementation_name = implementation.version_name + " (coloringBox " + box_dims + ")";
    // std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * a_d;
    DataType * b_d;
    DataType * x_d;
    DataType * y_d;
    DataType * result_d;

    CHECK_CUDA(hipMalloc(&a_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&b_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&result_d, sizeof(DataType)));

    CHECK_CUDA(hipMemcpy(a_d, a.data(), num_rows * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(b_d, b.data(), num_rows * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(x_d, x.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), num_rows * sizeof(DataType), hipMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    // we don't want to bench the implementation for the SymGS
    // (yet, because that has a bunch of parameters, specific to that, so we do an extra call to bench_SymGS)
    implementation.SymGS_implemented = false;
    bench_Implementation(implementation, *timer, striped_A, a_d, b_d, x_d, y_d, result_d, 1.0, 1.0);

    // free the memory
    hipFree(a_d);
    hipFree(b_d);
    hipFree(x_d);
    hipFree(y_d);
    hipFree(result_d);

    delete timer;

    // run the SymGS benchmark
    run_striped_COR_box_coloring_3d27p_SymGS_benchmark(nx, ny, nz, folder_path, implementation);
}
