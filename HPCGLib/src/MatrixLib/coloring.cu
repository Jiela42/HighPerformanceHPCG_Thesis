#include "MatrixLib/coloring.cuh"
#include "UtilLib/utils.cuh"
#include "UtilLib/cuda_utils.hpp"
// #include "banded_Matrix.hpp"

__global__ void color_for_forward_pass_kernel(
    int num_rows, int num_bands, int diag_offset, double * A, int * j_min_i, int * colors
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // we initialize the first row to be colored with 0
    if(tid == 0){
        colors[0] = 0;
    }
    
    // we loop over all the rows
    for(int i = tid; i < num_rows; i += gridDim.x * blockDim.x){
        int iterations_ctr = 0;
        
        int my_color = colors[i];
        // as long as our row is not colored we loop
        while(my_color < 0){

        // printf("from thread %d, block %d, i = %d, my_color %d \n", threadIdx.x, blockIdx.x, i, my_color);
            iterations_ctr++;
            int max_color = -1;
            int min_color = 0;

            // we loop over all the bands, checking if we can color the row
            for(int band = 0; band < diag_offset; band++){
                int j = j_min_i[band] + i;
                double val = A[i * num_bands + band];

                if(j < num_rows && j >= 0 && val != 0.0){
                    int color = colors[j];
                    
                    if(color > max_color){
                        max_color = color;
                    }
                    if(color < min_color){
                        min_color = color;
                    }
                }
            }

            if (min_color >= 0){
                // this means we can color the row
                my_color = max_color + 1;
                atomicExch(&colors[i], my_color);
                // printf("from thread %d, block %d, i = %d, my_color %d \n", threadIdx.x, blockIdx.x, i, my_color);
            }
        }
    }
}



std::vector<int> color_for_forward_pass(banded_Matrix <double> A){

    int num_rows = A.get_num_rows();
    int num_bands = A.get_num_bands();
    int diag_offset = A.get_diag_index();

    std::vector<int> colors(num_rows, -1);

    // put everything on the device
    int * colors_d;
    int * j_min_i_d;
    double * A_d;

    CHECK_CUDA(cudaMalloc(&colors_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_d, num_bands * num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemset(colors_d, -1, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, A.get_j_min_i().data(), num_rows * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_d, A.get_values().data(), num_bands * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    

    int num_threads = 1024;
    int num_blocks = 1;

    color_for_forward_pass_kernel<<<num_blocks, num_threads>>>(num_rows, num_bands, diag_offset, A_d, j_min_i_d, colors_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(colors.data(), colors_d, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(colors_d));
    CHECK_CUDA(cudaFree(j_min_i_d));
    CHECK_CUDA(cudaFree(A_d));

    return colors;

}