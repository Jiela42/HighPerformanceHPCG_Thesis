#include "testing.hpp"

#include "UtilLib/cuda_utils.hpp"

bool run_striped_warp_reduction_tests_on_matrix(sparse_CSR_Matrix<DataType>& A){
    // the output will be allocated by the test function
    // but any inputs need to be allocated and copied over to the device here
    // and is then passed to the test function

    bool all_pass = true;
    
    // create the baseline and the UUT
    cuSparse_Implementation<DataType> cuSparse;
    striped_warp_reduction_Implementation<DataType> striped_warp_reduction;
    
    int nx = A.get_nx();
    int ny = A.get_ny();
    int nz = A.get_nz();
    
    // random seeded x vector
    std::vector<DataType> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    striped_Matrix<DataType>* A_striped = A.get_Striped();

    // std::cout << "size: " << nx << "x" << ny << "x" << nz << std::endl;

    // // grab the j_min_i
    // std::vector<int> j_min_i = A_striped.get_j_min_i();

    // std::cout << "j_min_i: ";
    // for(int i = 0; i < j_min_i.size(); i++){
    //     std::cout << j_min_i[i] << " ";
    // }
    // std::cout << std::endl;

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();
    local_int_t nnz = A.get_nnz();

    DataType * a_d;
    DataType * b_d;
    DataType * x_d;
    DataType * y_d;


    // Allocate the memory on the device
    CHECK_CUDA(cudaMalloc(&a_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&b_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&y_d, num_cols * sizeof(DataType)));

    // Copy the data to the device    
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_cols * sizeof(DataType), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(DataType)));

    // test the SPMV function
    all_pass = all_pass && test_SPMV(
        cuSparse, striped_warp_reduction,
        *A_striped,
        a_d
        );


    // test the Dot function
    all_pass = all_pass && test_Dot(
        striped_warp_reduction,
        A_striped->get_nx(), A_striped->get_ny(), A_striped->get_nz()
        );

    
    // test the SymGS function (minitest, does not work with striped matrices)
    all_pass = all_pass && test_SymGS(
        cuSparse, striped_warp_reduction,
        *A_striped,
        y_d
    );

    all_pass = all_pass && test_WAXPBY(
        striped_warp_reduction,
        *A_striped,
        a_d, b_d
    );
        
    if(not all_pass){
        std::cout << "striped_warp_reduction tests failed for HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }
    
    // anything that got allocated also needs to be de-allocted

    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(a_d));
    CHECK_CUDA(cudaFree(b_d));

    return all_pass;
}

bool run_stripedWarpReduction_filebased_tests(){

    bool all_pass = true;

    striped_warp_reduction_Implementation<DataType> striped_warp_reduction;

    // MG tests
    all_pass = all_pass && test_MG(striped_warp_reduction);

    // CG tests
    all_pass = all_pass && test_CG(striped_warp_reduction);

    return all_pass;

}

bool run_stripedWarpReduction_tests(int nx, int ny, int nz){

    bool all_pass = true;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // standard 3d27pt matrix tests
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    


    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    sparse_CSR_Matrix <DataType>* current_matrix = &A;

    for(int i = 0; i < 3; i++){
        int current_nx = current_matrix->get_nx();
        int current_ny = current_matrix->get_ny();
        int current_nz = current_matrix->get_nz();

        if (current_nx % 2 == 0 and current_ny % 2 == 0 and current_nz % 2 == 0 and current_nx / 2 > 2 and current_ny / 2 > 2 and current_nz / 2 > 2){
            current_matrix->initialize_coarse_Matrix();
            current_matrix = current_matrix->get_coarse_Matrix();
        }
    }
    // std::cout << "we run the tests" << std::endl;

    all_pass = all_pass && run_striped_warp_reduction_tests_on_matrix(A);

    if(not all_pass){
        std::cout << "striped_warp_reduction tests failed for standard HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    }

    // these fancy tests only work on things like matrix-vector multiplication
    // A.iterative_values();

    // all_pass = all_pass && run_striped_warp_reduction_tests_on_matrix(A);
    // if(not all_pass){
    //     std::cout << "striped_warp_reduction tests failed for iterative values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    // }

    // A.random_values(RANDOM_SEED);
    // all_pass = all_pass && run_striped_warp_reduction_tests_on_matrix(A);
    // if(not all_pass){
    //     std::cout << "striped_warp_reduction tests failed for random values HPCG Matrix and size " << nx << "x" << ny << "x" << nz << std::endl;
    // }

    return all_pass;
   
}