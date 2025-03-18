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
    int max_color = (nx-1) + 2*(ny-1) + 4*(nz-1);

    // each thread will count the number of rows with a specific color

    for(int color = tid; color <= max_color; color += gridDim.x * blockDim.x){
        int my_num_rows = 0;

        // to search for the rows with a specific color we can do some math
        // ix_start = color%2, step = 2, end = nx
        // iy_start = ((color - ix) % 4) / 2, step, end = ny
        // for(int xy = color % 2; xy < nx * ny / 4 + color % 2; xy++){
        for(int xy = 0; xy < nx * ny; xy++){
            int x = xy / ny;
            int y = xy % ny;

            int enumerator = color - x - 2*y;

            // if (color == 16){
            //     printf("tid %d, x %d, y %d, enumerator %d \n",tid, x, y, enumerator);
            // }
            // these cases should not happen, we leave them here for debugging (for now)
            if(enumerator < 0){
                // printf("Error enumerator is negative, color %d, x %d, y %d\n", color, x, y);
                continue;
            }
            if(enumerator % 4 != 0){
                // printf("Error enumerator is not divisible by 4, color %d, x %d, y %d\n", color, x, y);
                continue;
            }

            int z = enumerator / 4;
            // if(color == 16){
            //     printf("color %d, x %d, y %d, z %d \n", color, x, y, z);
            // }

            if (z < nz){
                my_num_rows++;
                // if (color == 16){
                // printf("color %d, x %d, y %d, z %d, my_num_rows %d \n", color, x, y, z, my_num_rows);
                // }
            }
            // else {
            //     break;
            // }
        }
        // this is because we are using a prefix sum and need the first element to be 0
        color_pointer[color+1] = my_num_rows;
    }
}

__global__ void set_color_pointer_kernel(
    int nx, int ny, int nz,
    int * color_pointer
){
    // this kernel is sequential, we could use a fancy prefix sum kernel, if I wanted to implement it
    // before this kernel each color_pointer[i] contains the number of rows with color i

    int max_color = (nx-1) + 2*(ny-1) + 4*(nz-1);
    int num_colors = max_color + 1;

    color_pointer[0] = 0;

    for(int i = 1; i <= num_colors; i++){
        // printf("color %d has %d rows\n", i, color_pointer[i]);
        color_pointer[i] = color_pointer[i] + color_pointer[i-1];
    }
}

__global__ void sort_rows_by_color_kernel(
    int nx, int ny, int nz,
    int * color_pointer, int * color_sorted_rows
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int max_color = (nx-1) + 2*(ny-1) + 4*(nz-1);


    // if(tid == 0){
    //     printf("Max color from sort rows by color kernel %d\n", max_color);
    //     printf("nx %d, ny %d, nz %d\n", nx, ny, nz);
    // }

    for(int color = tid; color <= max_color; color += gridDim.x * blockDim.x){
        int my_color_ptr = color_pointer[color];

        // to search for the rows with a specific color we can do some math
        for(int xy = 0; xy < nx * ny; xy++){
            int x = xy / ny;
            int y = xy % ny;

            // if(my_color_ptr == 3){
            //     printf("I am thread %d, block %d, color %d, x %d, y %d\n", threadIdx.x, blockIdx.x, color, x, y);
            // }

            // if(my_color_ptr >= color_pointer[color+1]){
            //     printf("Error my_color pointer %d is bigger than the next color pointer %d, thread %d, color %d, x %d, y %d\n",my_color_ptr, color_pointer[color+1], threadIdx.x, color, x, y);
            //     break;
            // }

            int enumerator = color - x - 2*y;

            if(enumerator < 0){
                continue;
            }
            if(enumerator % 4 != 0){
                continue;
            }

            int z = enumerator / 4;

            if (z < nz){
                int row = x + y * nx + z * nx * ny;
                color_sorted_rows[my_color_ptr] = row;
                
                // if(my_color_ptr == 3){
                //     printf("I am thread %d, block %d, color %d, x %d, y %d, z %d, row %d\n", threadIdx.x, blockIdx.x, color, x, y, z, row);
                // }

                my_color_ptr++;
            }
        }
    }
}

__global__ void print_COR_Format_kernel(
    int max_colors, int num_rows, int * color_pointer, int * color_sorted_rows
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0){
        printf("COLOR POINTER\n");
        for(int i = 0; i <= max_colors; i++){
            printf("%d\n", color_pointer[i]);
        }
        // printf("\n");
        printf("COLOR SORTED ROWS\n");
        for(int i = 0; i < num_rows; i++){
            printf("%d\n", color_sorted_rows[i]);
        }
    }
}

__global__ void generate_COR_BoxColoring_kernel(
    int nx, int ny, int nz,
    int bx, int by, int bz,
    int * color_pointer, int * color_sorted_rows
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = tid; i < nx * ny * nz; i += gridDim.x * blockDim.x){
        
        // figure out the coordinates of the node
        int x = i % nx;
        int y = (i / nx) % ny;
        int z = i / (nx * ny);

        int mod_x = x % bx;
        int mod_y = y % by;
        int mod_z = z % bz;

        int color = mod_x + bx * mod_y + bx * by * mod_z;

        // now calculate how many nodes smaller i have the same color
        int num_color_cols_i = x / bx;
        int num_color_rows_i = y / by;
        int num_color_faces_i = z / bz;

        int num_color_cols_total = nx / bx;
        int num_color_rows_total = ny / by;
        int num_color_faces_total = nz / bz;

        int color_offs_x = color % bx;
        int color_offs_y = (color - color_offs_x) % (bx * by) / bx;
        int color_offs_z = (color - color_offs_x - bx * color_offs_y) / (bx * by);
        
        // the following adjustment on the other hand is absolutely necessary
        num_color_cols_total = (color_offs_x < nx % bx) ? (num_color_cols_total + 1) : num_color_cols_total;
        num_color_rows_total = (color_offs_y < ny % by) ? (num_color_rows_total + 1) : num_color_rows_total;
        num_color_faces_total = (color_offs_z < nz % bz) ? (num_color_faces_total + 1) : num_color_faces_total;

        int num_nodes_with_color_until_i = num_color_cols_i + num_color_rows_i * num_color_cols_total + num_color_faces_i * num_color_cols_total * num_color_rows_total;

        int my_row_location = color_pointer[color] + num_nodes_with_color_until_i;

        color_sorted_rows[my_row_location] = i;

        // if (color == 0){
        //    printf("i %d, x %d, y %d, z %d, color %d, num_color_cols_total %d, num_color_rows_total %d, num_color_faces_total %d,  my_row_location %d\n", i, x, y, z, color, num_color_cols_total, num_color_rows_total, num_color_faces_total, my_row_location);
        // }

    }
}

std::vector<int> color_for_forward_pass(striped_Matrix <double>& A){

    int num_rows = A.get_num_rows();
    int num_stripes = A.get_num_stripes();
    int diag_offset = A.get_diag_index();

    std::vector<int> colors(num_rows, -1);

    // put the matrix on the device (if not already there)
    if (A.get_values_d() == nullptr or A.get_j_min_i_d() == nullptr){
        A.copy_Matrix_toGPU();
    }
    
    double * A_d = A.get_values_d();
    int * j_min_i_d = A.get_j_min_i_d();

    // put colors on the device
    int * colors_d;
    CHECK_CUDA(cudaMalloc(&colors_d, num_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(colors_d, -1, num_rows * sizeof(int)));
    

    int num_threads = 1024;
    int num_blocks = 1;

    color_for_forward_pass_kernel<<<num_blocks, num_threads>>>(num_rows, num_stripes, diag_offset, A_d, j_min_i_d, colors_d);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(colors.data(), colors_d, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(colors_d));
    return colors;

}

std::vector <int> color_for_backward_pass(striped_Matrix <double>& A){

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

    int max_color = (nx-1) + 2*(ny-1) + 4*(nz-1);

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
    
    int max_color = (nx-1) + 2*(ny-1) + 4*(nz-1);
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

void print_COR_Format(int max_colors, int num_rows, int* color_pointer, int* color_sorted_rows){
    printf("Max colors: %d\n", max_colors);
    printf("Num rows: %d\n", num_rows);

    print_COR_Format_kernel<<<1,1>>>(max_colors, num_rows, color_pointer, color_sorted_rows);
}

void get_color_row_mapping_for_boxColoring(
    int nx,
    int ny,
    int nz,
    int *color_pointer_d,
    int * color_sorted_rows_d
){
    int num_colors = 27;
    int bx = 3;
    int by = 3;
    int bz = 3;
    
    std::vector<int> color_ptr_h(num_colors+1, 0);
    
    
    // because we have only 27 colors, we can do the color-setting sequentially
    for(int i = 1; i <= num_colors; i++){

        int color = i-1;

        int num_color_cols = nx / bx;
        int num_color_rows = ny / by;
        int num_color_faces = nz / bz;
        
        int color_offs_x = color % bx;
        int color_offs_y = (color - color_offs_x) % (bx * by) / bx;
        int color_offs_z = (color - color_offs_x - bx * color_offs_y) / (bx * by);

        num_color_cols = (color_offs_x < nx % bx) ? (num_color_cols + 1) : num_color_cols;
        num_color_rows = (color_offs_y < ny % by) ? (num_color_rows + 1) : num_color_rows;
        num_color_faces = (color_offs_z < nz % bz) ? (num_color_faces + 1) : num_color_faces;

        int num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces;

        color_ptr_h[i] = color_ptr_h[i-1] + num_nodes_with_color;
    }
    
    // copy the color pointer to the device
    CHECK_CUDA(cudaMemcpy(color_pointer_d, color_ptr_h.data(), (num_colors+1) * sizeof(int), cudaMemcpyHostToDevice));
    
    int num_threads = 1024;
    int num_blocks = std::min(ceiling_division(num_colors, num_threads), MAX_NUM_BLOCKS);

    generate_COR_BoxColoring_kernel<<<num_blocks, num_threads>>>(nx, ny, nz, bx, by, bz, color_pointer_d, color_sorted_rows_d);
    cudaDeviceSynchronize();

    // print_COR_Format(num_colors, nx * ny * nz, color_pointer_d, color_sorted_rows_d);

}
