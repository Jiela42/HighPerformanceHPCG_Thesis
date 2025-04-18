#include "benchmark.hpp"


void run_striped_coloring_3d27p_benchmarks(int nx, int ny, int nz, std::string folder_path, striped_coloring_Implementation<DataType>& implementation){

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

    // because this benchmark also need to bench the coloring, we need to delete and re-generate the matrix between runs

    int num_iterations = implementation.getNumberOfIterations();
    implementation.setNumberOfIterations(1);

    for(int iterations = 0; iterations < num_iterations; iterations++){
        // run the benchmarks (without the copying back and forth)

        int num_sanity_elements = 100;
        std::vector<DataType> a_original(num_sanity_elements);
        std::vector<DataType> b_original(num_sanity_elements);
        std::vector<DataType> x_original(num_sanity_elements);
        std::vector<DataType> y_original(num_sanity_elements);
    
        // copy the first 100 elements of the vectors
        CHECK_CUDA(cudaMemcpy(a_original.data(), a_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(b_original.data(), b_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(x_original.data(), x_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(y_original.data(), y_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
        
        // make a vector of vectors
        std::vector<std::vector<DataType>> original_vectors = {a_original, b_original, x_original, y_original};
        std::vector<DataType*> vectors_d = {a_d, b_d, x_d, y_d};
        
        // we do one sanity check prior to the benchmarking (just for my sanity and to make debugging easier)
        sanity_check_vectors(vectors_d, original_vectors);
    
        std::cout << "Bench SymGS" << std::endl;
        if(implementation.SymGS_implemented){
            bench_SymGS(implementation, *timer, striped_A, x_d, y_d);
            sanity_check_vectors(vectors_d, original_vectors);
            // delete A & re-generate it
            striped_A.~striped_Matrix();
            new (&striped_A) striped_Matrix<DataType>();
            striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);
    
            if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
                // initialize the MG data
                striped_Matrix <DataType>* current_matrix = &striped_A;
                for(int i = 0; i < 3; i++){
                    current_matrix->initialize_coarse_Matrix();
                    current_matrix = current_matrix->get_coarse_Matrix();
                }
            }
        }


        std::cout << "Bench CG" << std::endl;
        if(implementation.CG_implemented){
            bench_CG(implementation, *timer, striped_A, x_d, y_d);
            sanity_check_vectors(vectors_d, original_vectors);
            // delete A & re-generate it
            striped_A.~striped_Matrix();
            new (&striped_A) striped_Matrix<DataType>();
            striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);
    
            if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
                // initialize the MG data
                striped_Matrix <DataType>* current_matrix = &striped_A;
                for(int i = 0; i < 3; i++){
                    current_matrix->initialize_coarse_Matrix();
                    current_matrix = current_matrix->get_coarse_Matrix();
                }
            }
        }
        std::cout << "Bench MG" << std::endl;
        if(implementation.MG_implemented){
            bench_MG(implementation, *timer, striped_A, x_d, y_d);
            sanity_check_vectors(vectors_d, original_vectors);
            // delete A & re-generate it
            striped_A.~striped_Matrix();
            new (&striped_A) striped_Matrix<DataType>();
            striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);
    
            if(nx % 8 == 0 && ny % 8 == 0 && nz % 8 == 0 && nx / 8 > 2 && ny / 8 > 2 && nz / 8 > 2){
                // initialize the MG data
                striped_Matrix <DataType>* current_matrix = &striped_A;
                for(int i = 0; i < 3; i++){
                    current_matrix->initialize_coarse_Matrix();
                    current_matrix = current_matrix->get_coarse_Matrix();
                }
            }
        }
        
    }
    
    
    // free the memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(result_d);

    // restore the number of iterations
    implementation.setNumberOfIterations(num_iterations);
    
    delete timer;
}

void run_striped_coloring_3d27p_SymGS_benchmark(int nx, int ny, int nz, std::string folder_path, striped_coloring_Implementation<DataType>& implementation){
    
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


    // const DataType * matrix_data = striped_A->get_values().data();
    // const int * j_min_i_data = striped_A->get_j_min_i().data();
    
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


    int num_iterations = implementation.getNumberOfIterations();
    implementation.setNumberOfIterations(1);

    for(int iteration = 0; iteration < num_iterations; iteration++){

        bench_SymGS(implementation, *timer, striped_A, x_d, y_d);

        // delete A & re-generate it
        striped_A.~striped_Matrix();
        new (&striped_A) striped_Matrix<DataType>();
        striped_A.Generate_striped_3D27P_Matrix_onGPU(nx, ny, nz);
    }


    // free the memory
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));

    // restore the number of iterations
    implementation.setNumberOfIterations(num_iterations);

    delete timer;
}