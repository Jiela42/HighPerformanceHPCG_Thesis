#ifndef HPCG_multiGPU_UTILS_CUH
#define HPCG_multiGPU_UTILS_CUH

#include "cuda_runtime.h"

typedef int local_int_t;
typedef int global_int_t;

using DataType = double;

#define MPIDataType MPI_DOUBLE

#define CHECK_MPI(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
      printf("Failed: MPI error %s:%d '%d'\n",        \
          __FILE__,__LINE__, e);   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

#define CHECK_NCCL(cmd) do {                         \
ncclResult_t r = cmd;                             \
if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
}                                                 \
} while(0)

struct Problem_STRUCT{
    int npx, npy, npz;
    local_int_t nx, ny, nz;
    int size, rank;
    cudaStream_t stream;
    global_int_t gnx, gny, gnz;
    int px, py, pz;
    global_int_t gi0;
    global_int_t gx0, gy0, gz0;
};
typedef struct Problem_STRUCT Problem;

struct Halo_STRUCT{
    int nx;
    int ny;
    int nz;
    int dimx;
    int dimy;
    int dimz;
    DataType *interior;
    DataType *x_d;

    // host buffers for send and receive, only needed for host-only (non-)blocking MPI
    DataType *north_send_buff_h;
    DataType *north_recv_buff_h;
    
    DataType *east_send_buff_h;
    DataType *east_recv_buff_h;
    
    DataType *south_send_buff_h;
    DataType *south_recv_buff_h;
    
    DataType *west_send_buff_h;
    DataType *west_recv_buff_h;
    
    DataType *ne_send_buff_h;
    DataType *ne_recv_buff_h;
    
    DataType *se_send_buff_h;
    DataType *se_recv_buff_h;
    
    DataType *sw_send_buff_h;
    DataType *sw_recv_buff_h;
    
    DataType *nw_send_buff_h;
    DataType *nw_recv_buff_h;
    
    DataType *front_send_buff_h;
    DataType *front_recv_buff_h;
    
    DataType *back_send_buff_h;
    DataType *back_recv_buff_h;
    
    DataType *front_north_send_buff_h;
    DataType *front_north_recv_buff_h;
    
    DataType *front_east_send_buff_h;
    DataType *front_east_recv_buff_h;
    
    DataType *front_south_send_buff_h;
    DataType *front_south_recv_buff_h;
    
    DataType *front_west_send_buff_h;
    DataType *front_west_recv_buff_h;
    
    DataType *back_north_send_buff_h;
    DataType *back_north_recv_buff_h;
    
    DataType *back_east_send_buff_h;
    DataType *back_east_recv_buff_h;
    
    DataType *back_south_send_buff_h;
    DataType *back_south_recv_buff_h;
    
    DataType *back_west_send_buff_h;
    DataType *back_west_recv_buff_h;

    DataType *front_ne_send_buff_h;
    DataType *front_ne_recv_buff_h;
    
    DataType *front_se_send_buff_h;
    DataType *front_se_recv_buff_h;
    
    DataType *front_sw_send_buff_h;
    DataType *front_sw_recv_buff_h;
    
    DataType *front_nw_send_buff_h;
    DataType *front_nw_recv_buff_h;
    
    DataType *back_ne_send_buff_h;
    DataType *back_ne_recv_buff_h;
    
    DataType *back_se_send_buff_h;
    DataType *back_se_recv_buff_h;
    
    DataType *back_sw_send_buff_h;
    DataType *back_sw_recv_buff_h;
    
    DataType *back_nw_send_buff_h;
    DataType *back_nw_recv_buff_h;

    // host buffers for send and receive, only needed for RDMA exchanges
    DataType *north_send_buff_d;
    DataType *north_recv_buff_d;
    
    DataType *east_send_buff_d;
    DataType *east_recv_buff_d;
    
    DataType *south_send_buff_d;
    DataType *south_recv_buff_d;
    
    DataType *west_send_buff_d;
    DataType *west_recv_buff_d;
    
    DataType *ne_send_buff_d;
    DataType *ne_recv_buff_d;
    
    DataType *se_send_buff_d;
    DataType *se_recv_buff_d;
    
    DataType *sw_send_buff_d;
    DataType *sw_recv_buff_d;
    
    DataType *nw_send_buff_d;
    DataType *nw_recv_buff_d;
    
    DataType *front_send_buff_d;
    DataType *front_recv_buff_d;
    
    DataType *back_send_buff_d;
    DataType *back_recv_buff_d;
    
    DataType *front_north_send_buff_d;
    DataType *front_north_recv_buff_d;
    
    DataType *front_east_send_buff_d;
    DataType *front_east_recv_buff_d;
    
    DataType *front_south_send_buff_d;
    DataType *front_south_recv_buff_d;
    
    DataType *front_west_send_buff_d;
    DataType *front_west_recv_buff_d;
    
    DataType *back_north_send_buff_d;
    DataType *back_north_recv_buff_d;
    
    DataType *back_east_send_buff_d;
    DataType *back_east_recv_buff_d;
    
    DataType *back_south_send_buff_d;
    DataType *back_south_recv_buff_d;
    
    DataType *back_west_send_buff_d;
    DataType *back_west_recv_buff_d;

    DataType *front_ne_send_buff_d;
    DataType *front_ne_recv_buff_d;
    
    DataType *front_se_send_buff_d;
    DataType *front_se_recv_buff_d;
    
    DataType *front_sw_send_buff_d;
    DataType *front_sw_recv_buff_d;
    
    DataType *front_nw_send_buff_d;
    DataType *front_nw_recv_buff_d;
    
    DataType *back_ne_send_buff_d;
    DataType *back_ne_recv_buff_d;
    
    DataType *back_se_send_buff_d;
    DataType *back_se_recv_buff_d;
    
    DataType *back_sw_send_buff_d;
    DataType *back_sw_recv_buff_d;
    
    DataType *back_nw_send_buff_d;
    DataType *back_nw_recv_buff_d;

};
typedef struct Halo_STRUCT Halo;

enum MPI_Tags {
    NORTH = 0,
    NE = 1,
    EAST = 2,
    SE = 3,
    SOUTH = 4,
    SW = 5,
    WEST = 6,
    NW = 7,
    FRONT = 8,
    FRONT_NORTH = 9,
    FRONT_EAST = 10,
    FRONT_SOUTH = 11,
    FRONT_WEST = 12,
    BACK = 13,
    BACK_NORTH = 14,
    BACK_EAST = 15,
    BACK_SOUTH = 16,
    BACK_WEST = 17,
    FRONT_NE = 18,
    FRONT_SE = 19,
    FRONT_SW = 20,
    FRONT_NW = 21,
    BACK_NE = 22,
    BACK_SE = 23,
    BACK_SW = 24,
    BACK_NW = 25
};

void InitGPU(Problem *problem);

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem);

void InitHaloMemGPU(Halo *halo, int nx, int ny, int nz);
void InitHaloMemCPU(Halo *halo, int nx, int ny, int nz);
void InitHalo(Halo *halo, int nx, int ny, int nz);
void FreeHaloGPU(Halo *halo);
void FreeHaloCPU(Halo *halo);
void FreeHalo(Halo *halo);

void InjectDataToHalo(Halo *halo, DataType *data);
void SetHaloZeroGPU(Halo *halo);
void SetHaloGlobalIndexGPU(Halo *halo, Problem *problem);
void SetHaloQuotientGlobalIndexGPU(Halo *halo, Problem *problem);
void SetHaloRandomGPU(Halo *halo, Problem *problem, int min, int max, int seed);
void PrintHalo(Halo *x_d);
bool IsHaloZero(Halo *x_d);

void SendResult(int rank_recv, Halo *x_d, Problem *problem);
void GatherResult(Halo *x_d, Problem *problem, DataType *result_h);

void GenerateStripedPartialMatrix(Problem *problem, DataType *A);
bool VerifyPartialMatrix(DataType *striped_A_local_h, DataType *striped_A_global_h, int num_stripes, Problem *problem);

void extract_horizontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz);
void extract_vertical_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz);
void extract_frontal_plane_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz);
void extract_edge_X_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int dimx, int dimy, int dimz);
void extract_edge_Y_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz);
void extract_edge_Z_from_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz);

void inject_horizontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Z, int dimx, int dimy, int dimz);
void inject_vertical_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int length_Z, int dimx, int dimy, int dimz);
void inject_frontal_plane_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int length_Y, int dimx, int dimy, int dimz);
void inject_edge_X_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_X, int dimx, int dimy, int dimz);
void inject_edge_Y_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Y, int dimx, int dimy, int dimz);
void inject_edge_Z_to_GPU(DataType *x_d, DataType *x_h, int x, int y, int z, int length_Z, int dimx, int dimy, int dimz);

#endif