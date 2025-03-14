#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "UtilLib/cuda_utils.hpp"
#include <testing.hpp>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <cassert>
#include <stdbool.h>

__inline__ __device__ global_int_t local_i_to_halo_i(
    int i, 
    int nx, int ny, int nz,
    local_int_t dimx, local_int_t dimy
    )
    {
        return dimx*(dimy+1) + 1 + (i % nx) + dimx*((i % (nx*ny)) / nx) + (dimx*dimy)*(i / (nx*ny));
}

__global__ void inject_data_to_halo_kernel(DataType *x_d, DataType *data, int nx, int ny, int nz, int dimx, int dimy){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x){
            int hi = local_i_to_halo_i(i, nx, ny, nz, dimx, dimy);
            x_d[hi] = data[i];
    }
}

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem){
    problem->npx = npx; //number of processes in x
    problem->npy = npy; //number of processes in y
    problem->npz = npz; //number of processes in z
    problem->nx = nx; //number of grid points of processes subdomain in x
    problem->ny = ny; //number of grid points of processes subdomain in y
    problem->nz = nz; //number of grid points of processes subdomain in z
    problem->size = size;
    problem->rank = rank;
    problem->gnx = npx * nx; //global number of grid points in x
    problem->gny = npy * ny; //global number of grid points in y
    problem->gnz = npz * nz; //global number of grid points in z
    assert(size == npx * npy * npz); //each subdomain of size nx * ny * nz must be assigned to a process
    int px = rank % npx; //x index for this process in process grid
    int py = (rank % (npx * npy)) / npx; //y index for this process in process grid
    int pz = rank / (npx * npy); //z index for this process in process grid
    problem->px = px;
    problem->py = py;
    problem->pz = pz;
    problem->gi0 = (px * nx) + (py * npx * nx * ny) + (pz * npx * npy * nx * ny * nz); //base global index for this rank in the npx by npy by npz point grid layed out in 1D
    problem->gx0 = px * nx; //base global x index for this rank in the npx by npy by npz point grid
    problem->gy0 = py * ny; //base global y index for this rank in the npx by npy by npz point grid
    problem->gz0 = pz * nz; //base global z index for this rank in the npx by npy by npz point grid
}

void InitHaloMemGPU(Halo *halo, int nx, int ny, int nz){
    int dimx = nx + 2;
    int dimy = ny + 2;
    int dimz = nz + 2;
    DataType *x_d;
    CHECK_CUDA(cudaMalloc(&x_d, dimx * dimy * dimz * sizeof(DataType)));
    halo->x_d = x_d;
    DataType *interior = x_d + dimx * dimy + dimx + 1;
    halo->nx = nx;
    halo->ny = ny;
    halo->nz = nz;
    halo->dimx = dimx;
    halo->dimy = dimy;
    halo->dimz = dimz;
    halo->interior = interior;
}

void SetHaloZeroGPU(Halo *halo){
    CHECK_CUDA(cudaMemset(halo->x_d, 0, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType)));
}

void InjectDataToHalo(Halo *halo, DataType *data){
    int n = halo->nx * halo->ny * halo->nz;
    int num_threads = 1024;
    int num_blocks = std::min(MAX_NUM_BLOCKS, ceiling_division(n, num_threads));
    inject_data_to_halo_kernel<<<num_blocks, num_threads>>>(halo->x_d, data, halo->nx, halo->ny, halo->nz, halo->dimx, halo->dimy);
}

//correctness verified
void SetHaloGlobalIndexGPU(Halo *halo, Problem *problem){
    DataType *x_h = (DataType*) malloc(halo->dimx * halo->dimy * halo->dimz * sizeof(DataType));
    for(int i=0; i<halo->dimx * halo->dimy * halo->dimz; i++){
        x_h[i] = 0;
    }
    DataType *write_addr = x_h+ halo->dimx * halo->dimy + halo->dimx + 1;
    int gi = problem->gi0;
    for(int i = 0; i<halo->nz; i++){
        for(int j = 0; j<halo->ny; j++){
            for(int l = 0; l<halo->nx; l++){
                *write_addr = gi;
                gi++;
                write_addr++;
            }
            write_addr += 2;
            gi = gi - halo->nx + problem->gnx;
        }
        write_addr += 2 * halo->dimx;
        gi = problem->gi0 + (i + 1) * problem->gnx * problem->gny;
    }
    CHECK_CUDA(cudaMemcpy(halo->x_d, x_h, halo->dimx * halo->dimy * halo->dimz * sizeof(DataType), cudaMemcpyHostToDevice));
}

void FreeHaloGPU(Halo *halo){
    CHECK_CUDA(cudaFree(halo->x_d));
}

void InitGPU(Problem *problem){
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount > 0);
    CHECK_CUDA(cudaSetDevice(problem->rank % deviceCount));
    //printf("Rank=%d:\t\t Set my device to device=%d, available=%d.\n", problem->rank, problem->rank % deviceCount, deviceCount);
}

void extract_horizontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h += length_X;
        x_d += dimx * dimy;
    }
}

void inject_horizontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx * dimy;
        x_h += length_X;
    }
}

void extract_vertical_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        for(int j = 0; j < length_Y; j++){
            CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
            x_h++;
            x_d += dimx;
        }
        x_d += dimx * (dimy - length_Y);
    }
}

void inject_vertical_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Z; i++){
        for(int j = 0; j < length_Y; j++){
            CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
            x_d += dimx;
            x_h ++;
        }
        x_d += dimx * (dimy - length_Y);
    }
}

void extract_frontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Y; i++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h += length_X;
        x_d += dimx;
    }
}

void inject_frontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int i = 0; i < length_Y; i++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx;
        x_h += length_X;
    }

}

void extract_edge_X_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    CHECK_CUDA(cudaMemcpy(x_h, x_d, length_X * sizeof(DataType), cudaMemcpyDeviceToHost));
}

void inject_edge_X_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    CHECK_CUDA(cudaMemcpy(x_d, x_h, length_X * sizeof(DataType), cudaMemcpyHostToDevice));
}

void extract_edge_Y_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Y; j++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h++;
        x_d += dimx;
    }
}

void inject_edge_Y_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Y; j++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx;
        x_h ++;
    }
}

void extract_edge_Z_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Z; j++){
        CHECK_CUDA(cudaMemcpy(x_h, x_d, 1 * sizeof(DataType), cudaMemcpyDeviceToHost));
        x_h++;
        x_d += dimx * dimy;
    }
}

void inject_edge_Z_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz){
    local_int_t k = x +  y * dimx + z * dimx * dimy;
    x_d += k;
    for(int j = 0; j < length_Z; j++){
        CHECK_CUDA(cudaMemcpy(x_d, x_h, 1 * sizeof(DataType), cudaMemcpyHostToDevice));
        x_d += dimx * dimy;
        x_h ++;
    }
}

//correctness verified
void SendResult(int rank_recv, Halo *x_d, Problem *problem){
    DataType *send_addr_d = x_d->interior;
    DataType *send_buf_h = (DataType*) malloc(problem->nx * sizeof(DataType));
    for(int i = 0; i < problem->nz; i++){
        for(int j = 0; j<problem->ny; j++){
            CHECK_CUDA(cudaMemcpy(send_buf_h, send_addr_d, problem->nx * sizeof(DataType), cudaMemcpyDeviceToHost));
            MPI_Send(send_buf_h, problem->nx, MPIDataType, rank_recv, 0, MPI_COMM_WORLD);
            send_addr_d += x_d->dimx;
        }
        send_addr_d += 2 * x_d->dimx;
    }
    free(send_buf_h);
}

//correctness verified
void GatherResult(Halo *x_d, Problem *problem, DataType *result_h){
    DataType *own_data_d = x_d->interior;
    local_int_t data_paket = 0;
    for(int i = 0; i<problem->gnz; i++){ // go through all gnz layers
        int pz_recv = i / problem->nz;
        for(int j = 0; j<problem->gny; j++){ // go through all gny rows
            int py_recv = j / problem->ny;
            for(int l = 0; l<problem->npx; l++){ // go thorugh all nx columns
                int px_recv = l;
                int rank_recv = pz_recv * problem->npx * problem->npy + py_recv * problem->npx + px_recv;
                if(px_recv == problem->px && py_recv == problem->py && pz_recv == problem->pz){ //gathering rank holds the data
                    CHECK_CUDA(cudaMemcpy(result_h, own_data_d, problem->nx * sizeof(DataType), cudaMemcpyDeviceToHost));
                    for(int k = 0; k<problem->nx; k++){
                    }
                    own_data_d += x_d->dimx;
                }else{
                    MPI_Recv(result_h, problem->nx, MPIDataType, rank_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                result_h += problem->nx;
            }
        }
        if(pz_recv == problem->pz){
            own_data_d += 2 * x_d->dimx;
        }
    }
}

void PrintHalo(Halo *x_d){
    DataType *x_h = (DataType*) malloc(x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(x_h, x_d->x_d, x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType), cudaMemcpyDeviceToHost));
    for(int i = 0; i < x_d->dimz; i++){
        for(int j = 0; j < x_d->dimy; j++){
            for(int k = 0; k < x_d->dimx; k++){
                if(k == x_d->dimx - 1){
                    printf("\t");
                }
                printf("%f ", x_h[i*x_d->dimy*x_d->dimx + j*x_d->dimx + k]);
                if(k == 0){
                    printf("\t");
                }
            }
            if(j==0 || j == x_d->dimy - 2){
                printf("\n");
            }
            printf("\n");
        }
        printf("---\n");
    }
    free(x_h);
}

void GenerateStripedPartialMatrix(Problem *problem, DataType *A){
    int nx = problem->nx;
    int ny = problem->ny;
    int nz = problem->nz;
    global_int_t gnx = problem->gnx; //global number of points in x
    global_int_t gny = problem->gny; //global number of points in y
    global_int_t gnz = problem->gnz; //global number of points in z
    global_int_t gi0 = problem->gi0; //global index of local (0,0,0)
    
    local_int_t num_rows = nx * ny * nz;
    local_int_t num_cols = nx * ny * nz;
    
    for(int iz = 0; iz < nz; iz++){
        for(int iy = 0; iy < ny; iy++){
            for(int ix = 0; ix < nx; ix++){

                int i = ix + nx * iy + nx * ny * iz;

                int gx = problem->gx0 + ix; //global x index
                int gy = problem->gy0 + iy; //global y index
                int gz = problem->gz0 + iz; //global z index
                
                for (int sz = -1; sz < 2; sz++){
                    for(int sy = -1; sy < 2; sy++){
                        for(int sx = -1; sx < 2; sx++){

                            if(gx + sx < 0 || gx + sx >= gnx ||
                                gy + sy < 0 || gy + sy >= gny ||
                                gz + sz < 0 || gz + sz >= gnz) {
                                    *A = 0.0;
                                    A++;
                                } else {
                                    if(sx == 0 && sy == 0 && sz == 0){
                                        *A = 26.0;
                                        A++;
                                    } else {
                                        *A = -1.0;
                                        A++;
                                    }
                                }
                        }
                    }
                }
            }
        }
    }
}

bool VerifyPartialMatrix(DataType *striped_A_local_h, DataType *striped_A_global_h, int num_stripes, Problem *problem){
    for(int i = 0; i<problem->nz; i++){
        int gi0 = problem->gi0 + i * problem->gnx * problem->gny;
        for(int j = 0; j<problem->ny; j++){
            for(int k = 0; k<problem->nx; k++){
                for(int l = 0; l<num_stripes; l++){
                    if(striped_A_local_h[(k + j*problem->nx + i*problem->nx*problem->ny)*num_stripes + l] != striped_A_global_h[(gi0 + j*problem->gnx + k)*num_stripes + l]){
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool IsHaloZero(Halo *x_d){
    DataType *x_h = (DataType*) malloc(x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType));
    CHECK_CUDA(cudaMemcpy(x_h, x_d->x_d, x_d->dimx * x_d->dimy * x_d->dimz * sizeof(DataType), cudaMemcpyDeviceToHost));
    //check front and back
    for(int i = 0; i < x_d->dimx * x_d->dimy; i++){
        if(x_h[i] != 0.0 || x_h[i + x_d->dimx * x_d->dimy * (x_d->dimz - 1)] != 0.0){
            return false;
        }
    }
    //check middle part
    x_h += x_d->dimx * x_d->dimy;
    for(int iz = 0; iz < x_d->dimz-2; iz++){
        for(int iy = 0; iy < x_d->dimy; iy++){
            for(int ix = 0; ix < x_d->dimx; ix++){
                if(ix == 0 || ix == x_d->dimx - 1){
                    if(x_h[ix + iy * x_d->dimx + iz * x_d->dimx * x_d->dimy] != 0.0){
                        printf("ix = %d, iy = %d, iz = %d dimx=%d, dimy=%d, dimz=%d\n", ix, iy, iz, x_d->dimx, x_d->dimy, x_d->dimz);
                        return false;
                    }
                }
                if(iy == 0 || iy == x_d->dimy - 1){
                    if(x_h[ix + iy * x_d->dimx + iz * x_d->dimx * x_d->dimy] != 0.0){
                        printf("ix = %d, iy = %d, iz = %d dimx=%d, dimy=%d, dimz=%d\n", ix, iy, iz, x_d->dimx, x_d->dimy, x_d->dimz);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}