
#include <testing.hpp>

int main(){
    // this is supposed to show you how to run any of the functions the HPCG Library provides
    // we use a striped verison in this example

    int nx = 64;
    int ny = 64;
    int nz = 64;

    // create a matrix
    sparse_CSR_Matrix<double> A;
    A.generateMatrix_onGPU(nx, ny, nz);

    // create the coarse matrices for the mg routines
    sparse_CSR_Matrix <double>* current_matrix = &A;
    for(int i = 0; i < 3; i++){
        current_matrix->initialize_coarse_Matrix();
        current_matrix = current_matrix->get_coarse_Matrix();
    }

    // get the striped matrix
    striped_Matrix<double>* A_striped = A.get_Striped();

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // create the rest of the inputs/outputs

    // random seeded x vector
    std::vector<double> a = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> b = generate_random_vector(nx*ny*nz, RANDOM_SEED);
    std::vector<double> y = generate_y_vector_for_HPCG_problem(nx, ny, nz);

    // alpha and beta for waxpby
    double alpha = 0.5;
    double beta = 0.5;

    // delcarations of variables for CG
    int n_iters;
    double normr;
    double normr0;


    // declare an allocate the memory on the device
    double * a_d;
    double * b_d;
    double * x_d;
    double * y_d;
    double * scalar_d;

    CHECK_CUDA(cudaMalloc(&a_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&scalar_d, sizeof(double)));

    // copy the data to the device
    CHECK_CUDA(cudaMemcpy(a_d, a.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), num_cols * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(double)));

    // create an instance of the version to run the functions on
    striped_warp_reduction_Implementation<double> implementation;

    // run the functions
    std::cout << "Running the functions for size " << nx << "x" << ny << "x" << nz << " this may take a while" << std::endl;

    implementation.compute_CG(*A_striped, y_d, x_d, n_iters, normr, normr0);
    // reset the x vector
    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(double)));
    implementation.compute_MG(*A_striped, y_d, x_d);
    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(double)));
    implementation.compute_SymGS(*A_striped, x_d, y_d);
    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(double)));
    implementation.compute_SPMV(*A_striped, a_d, x_d);
    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(double)));
    implementation.compute_WAXPBY(*A_striped, a_d, b_d, x_d, alpha, beta);
    CHECK_CUDA(cudaMemset(x_d, 0, num_cols * sizeof(double)));
    implementation.compute_Dot(*A_striped, a_d, b_d, scalar_d);

    // free the memory
    CHECK_CUDA(cudaFree(a_d));
    CHECK_CUDA(cudaFree(b_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(y_d));
    CHECK_CUDA(cudaFree(scalar_d));

    std::cout << "The functions ran successfully" << std::endl;

    return 0;
}