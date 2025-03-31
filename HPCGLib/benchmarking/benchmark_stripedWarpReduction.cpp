#include "benchmark.hpp"


void run_striped_warp_reduction_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){

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

    srand(RANDOM_SEED);

    DataType alpha = (DataType)rand() / RAND_MAX;
    DataType beta = (DataType)rand() / RAND_MAX;

    if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
        // initialize the MG data
        striped_Matrix <DataType>* current_matrix = &striped_A;
        for(int i = 0; i < 3; i++){
            current_matrix->initialize_coarse_matrix();
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    }

    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

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

    // lets look at y_d
    // std::vector<DataType> y_host(5);
    // CHECK_CUDA(cudaMemcpy(y_host.data(), y_d, 5 * sizeof(DataType), cudaMemcpyDeviceToHost));

    // // print it
    // for(int i = 0; i < 5; i++){
    //     std::cout << y_host[i] << " ";
    // }

    // std::cout << std::endl;
    // print the address of y_d
    // std::cout << "address of y_d: " << y_d << std::endl;

    // run the benchmarks (without the copying back and forth)
    bench_Implementation(implementation, *timer, striped_A, a_d, b_d, x_d, y_d, result_d, alpha, beta);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
    // delete striped_A;
}

void run_warp_reduction_3d27p_Dot_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    // get two random vectors

    local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);

    std::vector<DataType> x = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> y = generate_random_vector(num_rows, RANDOM_SEED);

    // create the striped matrix
    striped_Matrix<DataType> A_striped;
    A_striped.set_num_rows(num_rows);

    // allocate x and y on the device
    DataType * x_d;
    DataType * y_d;
    DataType * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_rows* sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    // for(int i = 8; i <= 8; i=i << 1){
        // std::cout << "Cooperation number = " << i << std::endl;

        std::string implementation_name = implementation.version_name;
        // implementation.dot_cooperation_number = i;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        // the dot product is a dense operation, since we are just working on two vectors
        local_int_t nnz = num_rows;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);
    
        // run the benchmark
        bench_Dot(implementation, *timer, A_striped, x_d, y_d, result_d);

        delete timer;    
    // }
    
    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);
}

void run_warp_reduction_3d27p_WAXPBY_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    
        // create the striped matrix
        striped_Matrix<DataType> A_striped;
        
        local_int_t num_rows = static_cast<local_int_t>(nx) * static_cast<local_int_t>(ny) * static_cast<local_int_t>(nz);
        A_striped.set_num_rows(num_rows);
        
        // get two random vectors
        std::vector<DataType> x = generate_random_vector(num_rows, RANDOM_SEED);
        std::vector<DataType> y = generate_random_vector(num_rows, RANDOM_SEED);

        srand(RANDOM_SEED);

        DataType alpha = (DataType)rand() / RAND_MAX;
        DataType beta = (DataType)rand() / RAND_MAX;
    
        // allocate x and y on the device
        DataType * x_d;
        DataType * y_d;
        DataType * w_d;


    
        CHECK_CUDA(cudaMalloc(&x_d, num_rows * sizeof(DataType)));
        CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));
        CHECK_CUDA(cudaMalloc(&w_d, num_rows * sizeof(DataType)));
    
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

        std::string implementation_name = implementation.version_name;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        // the WAXBPY product is a dense operation, since we are just working on two vectors
        local_int_t nnz = num_rows;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);
    
        // run the benchmark
        bench_WAXPBY(implementation, *timer, A_striped, x_d, y_d, w_d, alpha, beta);

        delete timer;    
        
        // free the memory
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(w_d);
}

void run_warp_reduction_3d27p_SPMV_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    
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

    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * x_d;
    DataType * y_d;
    DataType * result_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));

    bench_SPMV(implementation, *timer, striped_A, x_d, y_d);
    
    // free the memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;
}

void run_warp_reduction_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_warp_reduction_Implementation<DataType>& implementation){
    
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

    
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    DataType * x_d;
    DataType * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(DataType)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(DataType), cudaMemcpyHostToDevice));

    bench_SymGS(implementation, *timer, striped_A, x_d, y_d);

    // free the memory
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    delete timer;
}