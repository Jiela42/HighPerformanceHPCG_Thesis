
#include <testing_hipified.hpp>

int main(){
    // this is supposed to show you how to run any of the functions the HPCG Library provides
    // we use a striped verison in this example

    int nx = 64;
    int ny = 64;
    int nz = 64;

    // create a matrix
    sparse_CSR_Matrix<DataType> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    // create the coarse matrices for the mg routines
    sparse_CSR_Matrix <DataType>* current_matrix = &A;
    for(int i = 0; i < 3; i++){
        current_matrix->initialize_coarse_Matrix();
        current_matrix = current_matrix->get_coarse_Matrix();
    }

    // get the striped matrix
    striped_Matrix<DataType>* A_striped = A.get_Striped();

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();

    // create the rest of the inputs/outputs

    // random seeded x vector
    std::vector<DataType> a = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> b = generate_random_vector(num_rows, RANDOM_SEED);
    std::vector<DataType> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    // alpha and beta for waxpby
    DataType alpha = 0.5;
    DataType beta = 0.5;

    // delcarations of variables for CG
    int n_iters;
    DataType normr;
    DataType normr0;


    // declare an allocate the memory on the device
    DataType * a_d;
    DataType * b_d;
    DataType * x_d;
    DataType * y_d;
    DataType * scalar_d;

    CHECK_CUDA(hipMalloc(&a_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&b_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&x_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&y_d, num_cols * sizeof(DataType)));
    CHECK_CUDA(hipMalloc(&scalar_d, sizeof(DataType)));

    // copy the data to the device
    CHECK_CUDA(hipMemcpy(a_d, a.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(b_d, b.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(y_d, y.data(), num_cols * sizeof(DataType), hipMemcpyHostToDevice));

    CHECK_CUDA(hipMemset(x_d, 0, num_cols * sizeof(DataType)));

    // create an instance of the version to run the functions on
    striped_warp_reduction_Implementation<DataType> implementation;

    // run the functions
    std::cout << "Running the functions for size " << nx << "x" << ny << "x" << nz << " this may take a while" << std::endl;

    implementation.compute_CG(*A_striped, y_d, x_d, n_iters, normr, normr0);
    // reset the x vector
    CHECK_CUDA(hipMemset(x_d, 0, num_cols * sizeof(DataType)));
    implementation.compute_MG(*A_striped, y_d, x_d);
    CHECK_CUDA(hipMemset(x_d, 0, num_cols * sizeof(DataType)));
    implementation.compute_SymGS(*A_striped, x_d, y_d);
    CHECK_CUDA(hipMemset(x_d, 0, num_cols * sizeof(DataType)));
    implementation.compute_SPMV(*A_striped, a_d, x_d);
    CHECK_CUDA(hipMemset(x_d, 0, num_cols * sizeof(DataType)));
    implementation.compute_WAXPBY(*A_striped, a_d, b_d, x_d, alpha, beta);
    CHECK_CUDA(hipMemset(x_d, 0, num_cols * sizeof(DataType)));
    implementation.compute_Dot(*A_striped, a_d, b_d, scalar_d);

    // free the memory
    CHECK_CUDA(hipFree(a_d));
    CHECK_CUDA(hipFree(b_d));
    CHECK_CUDA(hipFree(x_d));
    CHECK_CUDA(hipFree(y_d));
    CHECK_CUDA(hipFree(scalar_d));

    std::cout << "The functions ran successfully" << std::endl;

    return 0;
}