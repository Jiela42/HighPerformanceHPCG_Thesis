#include <cublas_v2.h>

#include <iostream>

void bench_cublas_matrix_vector(*A, b, x, bench_timer){

    for (int i = 0; i < bench_timer.num_iterations; i++){
        
        // allocate memory on device
        float *A_dev, *y_dev, *x_dev;
        CUDA_CHECK(cudaMalloc(&A, n * m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_dev, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&x_dev, m * sizeof(float)));

        // copy Av and y to device
        CUDA_CHECK(cudaMemcpy(A_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_dev, y, n * sizeof(float), cudaMemcpyHostToDevice));

        // call cublas matrix-vector multiplication
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDA_CHECK(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, A_dev, n, b_dev, 1, &beta, x_dev, 1);)

        // copy x back to host
        CUDA_CHECK(cudaMemcpy(x, x_dev, m * sizeof(float), cudaMemcpyDeviceToHost));

        // free memory on device
        CUDA_CHECK(cudaFree(A_dev));
        CUDA_CHECK(cudaFree(b_dev));
        CUDA_CHECK(cudaFree(x_dev));

        CUBLAS_CHECK(cublasDestroy(handle));

    }
}

void bench_for_reference() {
    // read matrix from file
    // read inverse matrix from file
    // read y vector from file

    // call matrix-vector multiplication cublas (A_inverse, n, m, y, x, bench_timer)

    // run an iterative gauss-seidel (single thread)
    // compare the results for correctness
    // run this a bunch of times to get a reasonable gauge of what is happening, safe results to csv file
}

