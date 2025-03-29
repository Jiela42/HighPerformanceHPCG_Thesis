#include "benchmark.hpp"


void run_striped_box_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<double>& implementation){
    
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);

    striped_Matrix<double> striped_A;
    striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);

    int num_rows = striped_A.get_num_rows();
    int num_cols = striped_A.get_num_cols();
    int nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();
    
    // Allocate the memory on the device
    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    for(int i = 3; i <= 3; i ++){
        
        implementation.bx = i;
        implementation.by = i;
        implementation.bz = i;

        std::string box_dims = std::to_string(implementation.bx) + "x" + std::to_string(implementation.by) + "x" + std::to_string(implementation.bz);
        std::string coop_num_string = std::to_string(implementation.SymGS_cooperation_number);
        
        // std::string implementation_name = implementation.version_name + " (coloringBox " + box_dims + ")" + " (coop_num " + coop_num_string + ")";
        std::string implementation_name = implementation.version_name;
        std::string additional_params = implementation.additional_parameters;
        std::string ault_node = implementation.ault_nodes;
        CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

        // std::cout << "Running the SymGS benchmark for the implementation: " << implementation_name << std::endl;

        bench_SymGS(implementation, *timer, striped_A, x_d, y_d);
    
        delete timer;
    }

    // free the memory
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

}

void run_striped_box_coloring_3d27p_CG_benchmark(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<double>& implementation){

    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    bool include_conversion = false;
    
    sparse_CSR_Matrix<double> A;
    striped_Matrix<double> striped_A;
    int num_rows;
    int nnz;
    if(include_conversion){

        A.generateMatrix_onGPU(nx, ny, nz);
        std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
        std::vector<double> x (nx*ny*nz, 0.0);
        std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
        std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    
        if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
            // initialize the MG data
            sparse_CSR_Matrix <double>* current_matrix = &A;
            for(int i = 0; i < 3; i++){
                current_matrix->initialize_coarse_Matrix();
                current_matrix = current_matrix->get_coarse_Matrix();
            }
        }

        num_rows = A.get_num_rows();
        nnz = A.get_nnz();
    } else {
        striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);

        std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
        std::vector<double> x (nx*ny*nz, 0.0);
        std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
        std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

        if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
            // initialize the MG data
            striped_Matrix <double>* current_matrix = &striped_A;
            for(int i = 0; i < 3; i++){
                current_matrix->initialize_coarse_matrix();
                current_matrix = current_matrix->get_coarse_Matrix();
            }
        }
        num_rows = striped_A.get_num_rows();
        nnz = striped_A.get_nnz();
    }

    // if we want to measure the time it takes CG, we now need to delete the data between runs (heul heul)

    std::string implementation_name = implementation.version_name;
    if(include_conversion){
        implementation_name += " (including conversion to Striped)";
    }

    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    double * x_d;
    double * y_d;

    CHECK_CUDA(cudaMalloc(&x_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    int num_iteration = include_conversion ? 10: 1;

    for(int i = 0; i< implementation.getNumberOfIterations(); i++){
        
        if(include_conversion){
            std::cout << "we include the conversion to striped in this run" << std::endl;
            bench_CG(
                implementation,
                *timer,
                A,
                x_d, y_d
            );
            // delete A & re-generate it
            A.~sparse_CSR_Matrix();
            new (&A) sparse_CSR_Matrix<double>();
            A.generateMatrix_onGPU(nx, ny, nz);

            if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
                // initialize the MG data
                sparse_CSR_Matrix <double>* current_matrix = &A;
                for(int i = 0; i < 3; i++){
                    current_matrix->initialize_coarse_Matrix();
                    current_matrix = current_matrix->get_coarse_Matrix();
                }
            }
        } else {
            // get the striped version and run that
            bench_CG(
                implementation,
                *timer,
                striped_A,
                x_d, y_d
            );
        }
    }

    // free da memory
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    delete timer;

}

void run_striped_box_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_box_coloring_Implementation<double>& implementation){

    striped_Matrix<double> striped_A;
    striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);

    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);
    std::vector<double> x (nx*ny*nz, 0.0);
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);

    if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
        // initialize the MG data
        striped_Matrix <double>* current_matrix = &striped_A;
        for(int i = 0; i < 3; i++){
            current_matrix->initialize_coarse_matrix();
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    }

    int num_rows = striped_A.get_num_rows();
    int num_cols = striped_A.get_num_cols();
    int nnz = striped_A.get_nnz();
    int num_stripes = striped_A.get_num_stripes();
    
    // std::string box_dims = std::to_string(implementation.bx) + "x" + std::to_string(implementation.by) + "x" + std::to_string(implementation.bz);
    // std::string implementation_name = implementation.version_name + "_box: " + box_dims;
    std::string implementation_name = implementation.version_name;
    std::string additional_params = implementation.additional_parameters;
    std::string ault_node = implementation.ault_nodes;
    CudaTimer* timer = new CudaTimer (nx, ny, nz, nnz, ault_node, "3d_27pt", implementation_name, additional_params, folder_path);

    // Allocate the memory on the device
    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;
    double * result_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_rows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&result_d, sizeof(double)));

    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice));

    // run the benchmarks (without the copying back and forth)
    // we don't want to bench the implementation for the SymGS
    // (yet, because that has a bunch of parameters, specific to that, so we do an extra call to bench_SymGS)
    implementation.SymGS_implemented = false;
    bench_Implementation(implementation, *timer, striped_A, a_d, b_d, x_d, y_d, result_d, 1.0, 1.0);

    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    delete timer;

    // run the SymGS benchmark
    run_striped_box_coloring_3d27p_SymGS_benchmark(nx, ny, nz, folder_path, implementation);
}
