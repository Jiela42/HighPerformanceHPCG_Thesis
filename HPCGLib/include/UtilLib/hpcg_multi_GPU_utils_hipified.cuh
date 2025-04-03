#ifndef HPCG_multiGPU_UTILS_CUH
#define HPCG_multiGPU_UTILS_CUH

#include "hip/hip_runtime.h"
#include "cuda_utils_hipified.hpp"

typedef long local_int_t;
typedef long global_int_t;

using DataType = double;

#define MPIDataType MPI_DOUBLE
#define NUMBER_NEIGHBORS 26

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

struct Halo_STRUCT;
typedef struct Halo_STRUCT Halo;

/*This is used for extraction and injection from a halo. E.g. extract a plane of size length_X x length_Y at point x, y, z, within halo of size dimx, dimy, dimz and interior size of nx, ny, nz*/
struct GhostCell_STRUCT{
    int x, y, z;
    int dimx, dimy, dimz;
    int length_X, length_Y, length_Z;
};
typedef struct GhostCell_STRUCT GhostCell;

struct Problem_STRUCT{
    int npx, npy, npz;
    local_int_t nx, ny, nz;
    int size, rank;
    hipStream_t stream;
    global_int_t gnx, gny, gnz;
    int px, py, pz;
    global_int_t gi0;
    global_int_t gx0, gy0, gz0;
    int neighbors[NUMBER_NEIGHBORS]; // stores the rank of the neighbors, follows the same order as Comm_Tags
    bool neighbors_mask[NUMBER_NEIGHBORS]; // stores if the neighbor is valid, follows the same order as Comm_Tags
    local_int_t count_exchange[NUMBER_NEIGHBORS]; // stores the number of elements to exchange with each neighbor, follows the same order as Comm_Tags
    GhostCell extraction_ghost_cells[NUMBER_NEIGHBORS]; // stores the geometry of the boundary with each neighbor for extraction, follows the same order as Comm_Tags
    GhostCell injection_ghost_cells[NUMBER_NEIGHBORS]; // stores the geometry of the boundary with each neighbor for injection, follows the same order as Comm_Tags
    void(*extraction_functions[NUMBER_NEIGHBORS])(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
    void(*injection_functions[NUMBER_NEIGHBORS])(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
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
    Problem *problem;

    DataType *send_buff_h[NUMBER_NEIGHBORS];
    DataType *recv_buff_h[NUMBER_NEIGHBORS];

    DataType *send_buff_d[NUMBER_NEIGHBORS];
    DataType *recv_buff_d[NUMBER_NEIGHBORS];

};
typedef struct Halo_STRUCT Halo;

enum Comm_Tags {
    NORTH = 0,
    EAST,
    SOUTH,
    WEST,
    NE,
    SE,
    SW,
    NW,
    FRONT,
    BACK,
    FRONT_NORTH,
    FRONT_EAST,
    FRONT_SOUTH,
    FRONT_WEST,
    BACK_NORTH,
    BACK_EAST,
    BACK_SOUTH,
    BACK_WEST,
    FRONT_NE,
    FRONT_SE,
    FRONT_SW,
    FRONT_NW,
    BACK_NE,
    BACK_SE,
    BACK_SW,
    BACK_NW
};

void InitGPU(Problem *problem);

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem);

void InitHaloMemGPU(Halo *halo, Problem *problem);
void InitHaloMemCPU(Halo *halo, Problem *problem);
void InitHalo(Halo *halo, Problem *problem);
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

void extract_horizontal_plane_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_horizontal_plane_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

void extract_vertical_plane_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_vertical_plane_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

void extract_frontal_plane_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_frontal_plane_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

void extract_edge_X_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_edge_X_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

void extract_edge_Y_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_edge_Y_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

void extract_edge_Z_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_edge_Z_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

void extract_corner_from_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);
void inject_corner_to_GPU(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);

#endif