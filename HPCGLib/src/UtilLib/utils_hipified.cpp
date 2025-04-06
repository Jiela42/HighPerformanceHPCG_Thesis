#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "UtilLib/utils_hipified.cuh"

#include "UtilLib/cuda_utils_hipified.hpp"
#include "UtilLib/hpcg_multi_GPU_utils_hipified.cuh"
#include "HPCG_versions/naiveStriped_hipified.cuh" // we need this for the matrix vector multiplication kernel
#include "HPCG_versions/striped_warp_reduction_hipified.cuh" // we need this for the matrix vector multiplication kernel

#include <cmath>

#include "UtilLib/utils_hipified.hpp"
#include "UtilLib/cuda_utils_hipified.hpp"
#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"

#include <cmath>

bool double_compare(double a, double b){

    if(std::isnan(a) or std::isnan(b)){
        std::cout << "One of the numbers is nan" << std::endl;
        return false;
    }
    if(std::abs(a - b) > error_tolerance){
        std::cout << "Error: " << a << " != " << b << std::endl;
        std::cout << "Difference: " << std::abs(a - b) << std::endl;
    }

    return std::abs(a - b) < error_tolerance;
}

bool relaxed_double_compare(double a, double b, double tolerance){
    // this function is for handling small issues like the kind that arise from exploiting commutativity on floats
    if(std::isnan(a) or std::isnan(b)){
        std::cout << "One of the numbers is nan" << std::endl;
        return false;
    }
    if(std::abs(a - b) >= tolerance){
        std::cout << "Error: " << a << " != " << b << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Difference: " << std::abs(a - b) << std::endl;
    }
    
    return std::abs(a - b) < tolerance;
}

bool vector_compare(const std::vector<double>& a, const std::vector<double>& b){
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match" << std::endl;
        return false;
    }
    int error_ctr = 0;
    for (int i = 0; i < a.size(); i++){
        if(std::isnan(a[i]) or std::isnan(b[i])){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << std::endl;
            return false;
        }
        if (not double_compare (a[i], b[i])){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename t>
bool vector_compare(const std::vector<t>& a, const std::vector<t>& b, std::string info){
    bool test_result = true;
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match for " << info << " size a = " << a.size() << " size b = " << b.size() << std::endl;
         test_result = false;
    }

    int fault_ctr = 0;

    for(local_int_t i = 0; i < a.size(); i++){
        if (a[i] != b[i] && fault_ctr < 10){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << " for " << info << std::endl;
            test_result = false;
            fault_ctr++;
        }
    }
    return test_result;
}

bool vector_compare(const std::vector<local_int_t>& a, const std::vector<local_int_t>& b, std::string info){
    bool test_result = true;
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match for " << info << " size a = " << a.size() << " size b = " << b.size() << std::endl;
         test_result = false;
    }

    int fault_ctr = 0;

    for(local_int_t i = 0; i < a.size(); i++){
        if (a[i] != b[i] && fault_ctr < 10){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << " for " << info << std::endl;
            test_result = false;
            fault_ctr++;
        }
    }
    return test_result;
}


double L2_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution){

    // this thing expects A, to be on the gpu

    std::vector<local_int_t> row_ptr = A.get_row_ptr();
    std::vector<local_int_t> col_idx = A.get_col_idx();
    std::vector<double> values = A.get_values();
    std::vector<double> Ax(x_solution.size(), 0.0);

    // first calculate Ax
    for (local_int_t i = 0; i < row_ptr.size() - 1; i++){
        for (local_int_t j = row_ptr[i]; j < row_ptr[i+1]; j++){
            Ax[i] += values[j] * x_solution[col_idx[j]];
        }
    }

    // now calculate the difference and the L2 norm
    double L2_norm = 0.0;

    for (local_int_t i = 0; i < x_solution.size(); i++){
        L2_norm += pow(Ax[i] - true_solution[i], 2);
    }

    return sqrt(L2_norm);
}

double relative_residual_norm_for_SymGS(sparse_CSR_Matrix<double>& A, std::vector<double> & x_solution, std::vector<double>& true_solution){
    std::vector<local_int_t> row_ptr = A.get_row_ptr();
    std::vector<local_int_t> col_idx = A.get_col_idx();
    std::vector<double> values = A.get_values();
    std::vector<double> Ax(x_solution.size(), 0.0);

    // First calculate Ax
    for (local_int_t i = 0; i < row_ptr.size() - 1; i++) {
        for (local_int_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            Ax[i] += values[j] * x_solution[col_idx[j]];
        }
    }

    // Now calculate the difference and the L2 norm
    double L2_norm = 0.0;
    for (local_int_t i = 0; i < x_solution.size(); i++) {
        L2_norm += pow(Ax[i] - true_solution[i], 2);
    }
    L2_norm = sqrt(L2_norm);

    // Calculate the L2 norm of the true solution
    double L2_norm_true = 0.0;
    for (local_int_t i = 0; i < true_solution.size(); i++) {
        L2_norm_true += pow(true_solution[i], 2);
    }
    L2_norm_true = sqrt(L2_norm_true);

    // Return the relative residual norm
    return L2_norm / L2_norm_true;
}

void sanity_check_vector(std::vector<double>& a, std::vector<double>& b){
    assert(a.size() == b.size());
    for (local_int_t i = 0; i < a.size(); i++){
        assert(double_compare(a[i], b[i]));
    }
}

void sanity_check_vectors(std::vector<double *>& device, std::vector<std::vector<double>>& original){
    assert(device.size() == original.size());

    for(local_int_t i = 0; i < device.size(); i++){
        // std::cout << "checking vector " << i << std::endl;
        std::vector<double> host(original[i].size());
        CHECK_CUDA(hipMemcpy(host.data(), device[i], original[i].size() * sizeof(double), hipMemcpyDeviceToHost));
        sanity_check_vector(host, original[i]);
    }
}

template bool vector_compare<int>(const std::vector<int>&, const std::vector<int>&, std::string);
template bool vector_compare<long>(const std::vector<long>&, const std::vector<long>&, std::string);
template bool vector_compare<long long>(const std::vector<long long>&, const std::vector<long long>&, std::string);


int ceiling_division(int numerator, int denominator) {
    return static_cast<int>(std::ceil(static_cast<double>(numerator) / denominator));
}

int next_smaller_power_of_two(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power / 2;
}

__global__ void matvec_mult_kernel(local_int_t num_rows, local_int_t *row_ptr, local_int_t *col_idx, double *values, double *x, double *Ax) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (local_int_t j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        Ax[row] = sum;
    }
}

__global__ void diff_and_norm_kernel(local_int_t num_rows, double *Ax, double *y, double *y_reduced) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    __shared__ double intermediate_sums[WARP_SIZE];

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        double d = Ax[row] - y[row];
        my_sum += d * d;
    }
    // reduce along warps
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(static_cast<unsigned long long>(0xFFFFFFFFFFFFFFFF),
        my_sum,
        static_cast<unsigned int>(offset),
        static_cast<int>(WARP_SIZE)
    );
    }

    __syncthreads();

    if (lane == 0) {
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // sync threads in the block
    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(static_cast<unsigned long long>(0xFFFFFFFFFFFFFFFF),
                my_sum,
                static_cast<unsigned int>(offset),
                static_cast<int>(WARP_SIZE)
            );
        }
    }

    if(threadIdx.x == 0){
        y_reduced[blockIdx.x] = my_sum;
        // printf("y_reduced[%d]: %f\n", blockIdx.x, y_reduced[blockIdx.x]);
    }
}

__global__ void square_fusedReduction_kernel(local_int_t num_rows, double *y, double *y_reduced) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    __shared__ double intermediate_sums[WARP_SIZE];

    // printf("Hello from the kernel\n");

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        my_sum += y[row] * y[row];
    }

    // reduce along warps
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(static_cast<unsigned long long>(0xFFFFFFFFFFFFFFFF),
        my_sum,
        static_cast<unsigned int>(offset),
        static_cast<int>(WARP_SIZE)
    );
    }

    __syncthreads();

    if (lane == 0) {
        intermediate_sums[warp_id] = my_sum;
    }

    __syncthreads();

    // sync threads in the block
    if(warp_id == 0){
        my_sum = intermediate_sums[lane];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
            my_sum += __shfl_down_sync(static_cast<unsigned long long>(0xFFFFFFFFFFFFFFFF),
                my_sum,
                static_cast<unsigned int>(offset),
                static_cast<int>(WARP_SIZE)
            );
        }
    }

    if(threadIdx.x == 0){
        y_reduced[blockIdx.x] = my_sum;
        // printf("y_reduced[%d]: %f\n", blockIdx.x, y_reduced[blockIdx.x]);
    }
}

__global__ void compute_restriction_kernel(
    local_int_t num_rows,
    DataType * Axf,
    DataType * rf,
    DataType * rc,
    local_int_t * f2c_operator
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        rc[row] = rf[f2c_operator[row]] - Axf[f2c_operator[row]];
    }
}

__inline__ __device__ global_int_t local_i_to_halo_i(
    int i,
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
    )
    {
        return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

__global__ void compute_restriction_multi_GPU_kernel(
    local_int_t num_rows,
    DataType * Axf, //halo access, fine number of rows
    DataType * rf, //halo access, fine number of rows
    DataType * rc, //halo access, coarse number of rows
    local_int_t *f2c_operator, //normal access
    int nx, int ny, int nz
){
    local_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = tid; row < num_rows; row += blockDim.x * gridDim.x) {
        local_int_t hi = local_i_to_halo_i(row, nx / 2, ny / 2, nz / 2, nx / 2 + 2, ny / 2 + 2); // bc we need to access rc and rc is coarse
        local_int_t f2c_hi = local_i_to_halo_i(f2c_operator[row], nx, ny, nz, nx + 2, ny + 2);
        rc[hi] = rf[f2c_hi] - Axf[f2c_hi];
    }
}


__global__ void compute_prolongation_kernel(
    local_int_t num_rows,
    double * xc,
    double * xf,
    local_int_t * f2c_operator
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x) {

        // if(f2c_operator[row] < 0){

        //     printf("f2c_operator is the problem for row %d\n", row);
        //     printf("f2c_operator[%d]: %d\n", row, f2c_operator[row]);
        //     printf("xc is the problem for row %d\n", row);
        //     printf("xc[row]: %f\n", xc[row]);
        //     printf("xf is the problem for row %d\n", row);
        //     printf("xf[f2c_operator[%d]]: %f\n",row, xf[f2c_operator[row]]);
        // }


        xf[f2c_operator[row]] += xc[row];
    }
}

__global__ void compute_prolongation_multi_GPU_kernel(
    local_int_t num_rows,
    DataType * xc, //coarse, Halo
    DataType * xf, //fine, Halo
    local_int_t * f2c_operator,
    int nx, int ny, int nz
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(local_int_t row = tid; row < num_rows; row += blockDim.x * gridDim.x) {

        // if(f2c_operator[row] < 0){

        //     printf("f2c_operator is the problem for row %d\n", row);
        //     printf("f2c_operator[%d]: %d\n", row, f2c_operator[row]);
        //     printf("xc is the problem for row %d\n", row);
        //     printf("xc[row]: %f\n", xc[row]);
        //     printf("xf is the problem for row %d\n", row);
        //     printf("xf[f2c_operator[%d]]: %f\n",row, xf[f2c_operator[row]]);
        // }

        local_int_t f2c_fine_hi = local_i_to_halo_i(f2c_operator[row], nx, ny, nz, nx + 2, ny + 2);
        local_int_t f2c_coarse_hi = local_i_to_halo_i(row, nx / 2, ny / 2, nz / 2, nx / 2 + 2, ny / 2 + 2);
        xf[f2c_fine_hi] += xc[f2c_coarse_hi];
    }
}

void L2_norm_for_Device_Vector(
    hipStream_t stream,
    local_int_t num_rows,
    double * y,
    double * result
){

    // Allocate memory for the squared vector on the device
    double *y_squared;

    // std::cout << "Hello from L2_norm_for_Device_Vector" << std::endl;


    // Compute the squared vector and start reducing
    int blockSize = 1024;
    int numBlocks = (num_rows + blockSize - 1) / (blockSize*8);
    // we need to make sure there is at least one block
    numBlocks = std::max(numBlocks, 1);
    CHECK_CUDA(hipMalloc(&y_squared, numBlocks * sizeof(double)));
    square_fusedReduction_kernel<<<numBlocks, blockSize, 0, stream>>>(num_rows, y, y_squared);

    // Use Thrust to compute the sum of the squared vector
    thrust::device_ptr<double> y_squared_ptr(y_squared);
    *result = std::sqrt(thrust::reduce(thrust::hip::par.on(stream), y_squared_ptr, y_squared_ptr + numBlocks));

    // Free device memory
    CHECK_CUDA(hipFreeAsync(y_squared, stream));

}

double L2_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
){

    local_int_t num_rows = A.get_num_rows();
    local_int_t * row_ptr = A.get_row_ptr_d();
    local_int_t * col_idx = A.get_col_idx_d();
    double * values = A.get_values_d();

    // Allocate memory for Ax on the device
    double *Ax;
    double *diff;
    CHECK_CUDA(hipMalloc(&Ax, num_rows * sizeof(double)));
    CHECK_CUDA(hipMalloc(&diff, num_rows * sizeof(double)));

    CHECK_CUDA(hipMemset(Ax, 0, num_rows * sizeof(double)));

    // Perform matrix-vector multiplication: Ax = A * x

    int blockSize = 1024;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;
    matvec_mult_kernel<<<numBlocks, blockSize>>>(num_rows, row_ptr, col_idx, values, x, Ax);
    CHECK_CUDA(hipDeviceSynchronize());
    // // print Ax
    // double *Ax_h = (double *)malloc(num_rows * sizeof(double));
    // CHECK_CUDA(hipMemcpy(Ax_h, Ax, num_rows * sizeof(double), hipMemcpyDeviceToHost));
    // for (int i = 0; i < num_rows; i++) {
    //     std::cout << Ax_h[i] << " ";
    // }
    // std::cout << std::endl;

    // Compute the difference and the squared differences
    diff_and_norm_kernel<<<numBlocks, blockSize>>>(num_rows, Ax, y, diff);
    CHECK_CUDA(hipDeviceSynchronize());

    // Use Thrust to compute the sum of the squared differences
    thrust::device_ptr<double> diff_ptr(diff);
    double L2_norm = std::sqrt(thrust::reduce(diff_ptr, diff_ptr + numBlocks));

    // Free device memory
    CHECK_CUDA(hipFree(Ax));
    CHECK_CUDA(hipFree(diff));

    return L2_norm;
}

double L2_norm_for_SymGS(
    striped_Matrix<double> & A,
    double * x,
    double * y
){

    striped_warp_reduction_Implementation<double> implementation;
    local_int_t num_rows = A.get_num_rows();

    // Allocate memory for Ax on the device
    double *Ax;
    double *result;
    CHECK_CUDA(hipMalloc(&Ax, num_rows * sizeof(double)));
    CHECK_CUDA(hipMalloc(&result, 1 * sizeof(double)));

    CHECK_CUDA(hipMemset(Ax, 0, num_rows * sizeof(double)));
    CHECK_CUDA(hipMemset(result, 0, 1 * sizeof(double)));

    // Perform matrix-vector multiplication: Ax = A * x

    implementation.compute_SPMV(A, x, Ax);
    implementation.compute_WAXPBY(A, Ax, y, Ax, 1.0, -1.0);
    implementation.compute_Dot(A, Ax, Ax, result);

    // copy result over
    double result_h;
    CHECK_CUDA(hipMemcpy(&result_h, result, 1 * sizeof(double), hipMemcpyDeviceToHost));


    // Free device memory
    CHECK_CUDA(hipFree(Ax));
    CHECK_CUDA(hipFree(result));

    return std::sqrt(result_h);
}


double relative_residual_norm_for_SymGS(
    sparse_CSR_Matrix<double> & A,
    double * x,
    double * y
){

    local_int_t num_rows = A.get_num_rows();
    local_int_t * row_ptr = A.get_row_ptr_d();
    local_int_t * col_idx = A.get_col_idx_d();
    double * values = A.get_values_d();


    // make new cuda stream for the vector y computation
    hipStream_t stream;
    CHECK_CUDA(hipStreamCreate(&stream));
    double L2_norm_y;

    L2_norm_for_Device_Vector(stream, num_rows, y, &L2_norm_y);
    // the order of the calls is important, because L2_norm_for_SymGS will synchronize the default stream and not return until it does
    double L2_norm = L2_norm_for_SymGS(A, x, y);

    CHECK_CUDA(hipStreamSynchronize(stream));
    CHECK_CUDA(hipStreamDestroy(stream));
    return L2_norm / L2_norm_y;
}

double relative_residual_norm_for_SymGS(
    striped_Matrix<double> & A,
    double * x,
    double * y
){

    local_int_t num_rows = A.get_num_rows();
    local_int_t num_cols = A.get_num_cols();

    int num_stripes = A.get_num_stripes();
    local_int_t * j_min_i = A.get_j_min_i_d();
    double * striped_A_d = A.get_values_d();

    // std::cout << "relative_residual_norm_for_SymGS" << std::endl;
    // make new cuda stream for the vector y computation
    hipStream_t stream;
    CHECK_CUDA(hipStreamCreate(&stream));

    double L2_norm_y;

    L2_norm_for_Device_Vector(stream, num_rows, y, &L2_norm_y);
    // the order of the calls is important, because L2_norm_for_SymGS will synchronize the default stream and not return until it does
    double L2_norm = L2_norm_for_SymGS(A, x, y);

    CHECK_CUDA(hipStreamSynchronize(stream));
    CHECK_CUDA(hipStreamDestroy(stream));
    CHECK_CUDA(hipDeviceSynchronize());
    // std::cout << "L2_norm: " << L2_norm << std::endl;
    // std::cout << "L2_norm_y: " << L2_norm_y << std::endl;
    return L2_norm / L2_norm_y;
}









