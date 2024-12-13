#include "MatrixLib/coloring.cuh"
#include "UtilLib/utils.cuh"
#include "UtilLib/cuda_utils.hpp"

__global__ void color_for_forward_pass_kernel(
    int num_rows, int num_stripes, int diag_offset, double * A, int * j_min_i, int * colors
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

            // we loop over all the stripes, checking if we can color the row
            for(int stripe = 0; stripe < diag_offset; stripe++){
                int j = j_min_i[stripe] + i;
                double val = A[i * num_stripes + stripe];

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

__global__ void color_for_backward_pass_kernel(
    int num_rows, int num_stripes, int diag_offset, double * A, int * j_min_i, int * colors
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // we initialize the last row to be colored with 0
    if(tid == 0){
        colors[num_rows-1] = 0;
    }
    
    // we loop over all the rows
    for(int i = num_rows - tid-1; i >= 0; i -= gridDim.x * blockDim.x){
        int iterations_ctr = 0;
        
        int my_color = colors[i];
        // as long as our row is not colored we loop
        while(my_color < 0){

        // printf("from thread %d, block %d, i = %d, my_color %d \n", threadIdx.x, blockIdx.x, i, my_color);
            iterations_ctr++;
            int max_color = -1;
            int min_color = 0;

            // we loop over all the stripes, checking if we can color the row
            for(int stripe = diag_offset + 1; stripe < num_stripes; stripe++){
                int j = j_min_i[stripe] + i;
                double val = A[i * num_stripes + stripe];

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

__global__ void count_num_row_per_color_kernel(
    int nx, int ny, int nz,
    int * color_pointer
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int max_color = nx + 2*ny + 4*nz;

    // each thread will count the number of rows with a specific color

    for(int color = tid; color <= max_color; color += gridDim.x * blockDim.x){
        int my_num_rows = 0;

        // to search for the rows with a specific color we can do some math
        for(int xy = 0; xy < nx * ny; xy++){
            int x = xy % ny;
            int y = xy / ny;

            int enumerator = color - x - 2*y;

            if(enumerator < 0){
                break;
            }
            if(enumerator % 4 != 0){
                continue;
            }

            int z = enumerator / 4;

            if (z < nz){
                my_num_rows++;
            } else {
                break;
            }
        }
        // this is because we are using a prefix sum and need the first element to be 0
        color_pointer[color + 1] = my_num_rows;
    }
}

__global__ void set_color_pointer_kernel(
    int nx, int ny, int nz,
    int * color_pointer
){
    // this kernel is sequential, we could use a fancy prefix sum kernel, if I wanted to implement it
    // before this kernel each color_pointer[i] contains the number of rows with color i

    int max_color = nx + 2*ny + 4*nz;

    for(int i = 1; i <= max_color; i++){
        color_pointer[i] = color_pointer[i] + color_pointer[i-1];
    }
}

__global__ void sort_rows_by_color_kernel(
    int nx, int ny, int nz,
    int * color_pointer, int * color_sorted_rows
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int max_color = nx + 2*ny + 4*nz;

    for(int color = tid; color <= max_color; color += gridDim.x * blockDim.x){
        int my_color_ptr = color_pointer[color];

        // to search for the rows with a specific color we can do some math
        for(int xy = 0; xy < nx * ny; xy++){
            int x = xy % ny;
            int y = xy / ny;

            int enumerator = color - x - 2*y;

            if(enumerator < 0){
                break;
            }
            if(enumerator % 4 != 0){
                continue;
            }

            int z = enumerator / 4;

            if (z < nz){
                int row = x + y * nx + z * nx * ny;
                color_sorted_rows[my_color_ptr] = row;
                my_color_ptr++;
            }
        }
    }
}

std::vector<int> color_for_forward_pass(striped_Matrix <double> A){

    int num_rows = A.get_num_rows();
    int num_stripes = A.get_num_stripes();
    int diag_offset = A.get_diag_index();

    std::vector<int> colors(num_rows, -1);

    // put everything on the device
    int * colors_d;
    int * j_min_i_d;
    double * A_d;

    CHECK_CUDA(cudaMalloc(&colors_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_d, num_stripes * num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemset(colors_d, -1, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, A.get_j_min_i().data(), num_rows * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_d, A.get_values().data(), num_stripes * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    

    int num_threads = 1024;
    int num_blocks = 1;

    color_for_forward_pass_kernel<<<num_blocks, num_threads>>>(num_rows, num_stripes, diag_offset, A_d, j_min_i_d, colors_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(colors.data(), colors_d, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(colors_d));
    CHECK_CUDA(cudaFree(j_min_i_d));
    CHECK_CUDA(cudaFree(A_d));

    return colors;

}

std::vector <int> color_for_backward_pass(striped_Matrix <double> A){

    int num_rows = A.get_num_rows();
    int num_stripes = A.get_num_stripes();
    int diag_offset = A.get_diag_index();

    std::vector<int> colors(num_rows, -1);

    // put everything on the device
    int * colors_d;
    int * j_min_i_d;
    double * A_d;

    CHECK_CUDA(cudaMalloc(&colors_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&j_min_i_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&A_d, num_stripes * num_rows * sizeof(double)));

    CHECK_CUDA(cudaMemset(colors_d, -1, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(j_min_i_d, A.get_j_min_i().data(), num_rows * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_d, A.get_values().data(), num_stripes * num_rows * sizeof(double), cudaMemcpyHostToDevice));
    

    int num_threads = 1024;
    int num_blocks = 1;

    color_for_backward_pass_kernel<<<num_blocks, num_threads>>>(num_rows, num_stripes, diag_offset, A_d, j_min_i_d, colors_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(colors.data(), colors_d, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(colors_d));
    CHECK_CUDA(cudaFree(j_min_i_d));
    CHECK_CUDA(cudaFree(A_d));

    return colors;

}


void get_color_row_mapping(int nx, int ny, int nz, int *color_pointer_d, int * color_sorted_rows_d){
    
    // first we find the number of rows per color
    // i.e. set the color_pointer_d

    int max_color = nx + 2*ny + 4*nz;

    int num_threads = 1024;
    int num_blocks = std::min(ceiling_division(max_color+1, num_threads), MAX_NUM_BLOCKS);

    count_num_row_per_color_kernel<<<num_blocks, num_threads>>>(nx, ny, nz, color_pointer_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    // sadly this next part is sequential. there is a fancy butterfly implementation, but I am lazy

    set_color_pointer_kernel<<<1, 1>>>(nx, ny, nz, color_pointer_d);
    
    CHECK_CUDA(cudaDeviceSynchronize());

    // now we sort the rows by color

    sort_rows_by_color_kernel<<<num_blocks, num_threads>>>(nx, ny, nz, color_pointer_d, color_sorted_rows_d);

    CHECK_CUDA(cudaDeviceSynchronize());

}

std::pair<std::vector<int>, std::vector<int>> get_color_row_mapping(int nx, int ny, int nz)
{
    
    int max_color = nx + 2*ny + 4*nz;
    int num_rows = nx * ny * nz;

    std::vector<int> color_pointer (max_color+1, 0);
    std::vector<int> color_sorted_rows (num_rows, -1);

    // allocate space for the vectors on the device
    int * color_pointer_d;
    int * color_sorted_rows_d;

    CHECK_CUDA(cudaMalloc(&color_pointer_d, (max_color+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&color_sorted_rows_d, num_rows * sizeof(int)));

    CHECK_CUDA(cudaMemset(color_pointer_d, 0, (max_color+1) * sizeof(int)));

    get_color_row_mapping(nx, ny, nz, color_pointer_d, color_sorted_rows_d);

    // copy the results back to the host
    CHECK_CUDA(cudaMemcpy(color_pointer.data(), color_pointer_d, (max_color+1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(color_sorted_rows.data(), color_sorted_rows_d, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    // free the memory
    CHECK_CUDA(cudaFree(color_pointer_d));
    CHECK_CUDA(cudaFree(color_sorted_rows_d));

    return std::make_pair(color_pointer, color_sorted_rows);

}