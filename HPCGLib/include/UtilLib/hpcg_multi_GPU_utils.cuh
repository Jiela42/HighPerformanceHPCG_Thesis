#ifndef HPCG_multiGPU_UTILS_CUH
#define HPCG_multiGPU_UTILS_CUH

// Utility definitions and function declarations for multi-GPU communication and data handling in HPCG

#include "cuda_runtime.h"
#include "cuda_utils.hpp"

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

// Forward declaration to allow use of Halo* in Problem_STRUCT
struct Halo_STRUCT;
typedef struct Halo_STRUCT Halo;

// Represents a subregion (ghost cell) of the halo used for communication between neighboring subdomains.
struct GhostCell_STRUCT{
    local_int_t x, y, z;               // Starting coordinates of the ghost cell within the halo
    local_int_t dimx, dimy, dimz;      // Dimensions of the overall halo region
    local_int_t length_X, length_Y, length_Z;  // Size of the ghost cell along each dimension
};
typedef struct GhostCell_STRUCT GhostCell;

// Describes the problem decomposition and communication metadata for a subdomain.
struct Problem_STRUCT{
    int npx, npy, npz;                // Number of subdomains in each dimension
    local_int_t nx, ny, nz;           // Local subdomain sizes
    int size, rank;                   // MPI size and rank
    cudaStream_t stream;              // CUDA stream for asynchronous operations
    global_int_t gnx, gny, gnz;       // Global problem sizes
    int px, py, pz;                  // Position of this subdomain in the grid
    global_int_t gi0;                // Global index offset
    global_int_t gx0, gy0, gz0;      // Global origin coordinates of the subdomain
    int neighbors[NUMBER_NEIGHBORS]; // Ranks of neighboring subdomains, ordered by Comm_Tags
    bool neighbors_mask[NUMBER_NEIGHBORS]; // Validity mask for neighbors, ordered by Comm_Tags
    local_int_t count_exchange[NUMBER_NEIGHBORS]; // Number of elements to exchange per neighbor
    GhostCell extraction_ghost_cells[NUMBER_NEIGHBORS]; // Geometry for extracting data to send
    GhostCell injection_ghost_cells[NUMBER_NEIGHBORS];  // Geometry for injecting received data
    void(*extraction_functions[NUMBER_NEIGHBORS])(Halo *halo, int i_buff, GhostCell *gh, bool host_buff); // Extraction function pointers
    void(*injection_functions[NUMBER_NEIGHBORS])(Halo *halo, int i_buff, GhostCell *gh, bool host_buff);  // Injection function pointers
};
typedef struct Problem_STRUCT Problem;

// Stores halo data used for communication between GPUs and associated metadata.
struct Halo_STRUCT{
    local_int_t nx;                 // Local subdomain size in x
    local_int_t ny;                 // Local subdomain size in y
    local_int_t nz;                 // Local subdomain size in z
    local_int_t dimx;               // Halo dimension in x including ghost layers
    local_int_t dimy;               // Halo dimension in y including ghost layers
    local_int_t dimz;               // Halo dimension in z including ghost layers
    DataType *interior;             // Pointer to interior data array
    DataType *x_d;                 // Device pointer to halo data
    Problem *problem;               // Associated problem metadata

    DataType *send_buff_h[NUMBER_NEIGHBORS]; // Host send buffers for each neighbor
    DataType *recv_buff_h[NUMBER_NEIGHBORS]; // Host receive buffers for each neighbor

    DataType *send_buff_d[NUMBER_NEIGHBORS]; // Device send buffers for each neighbor
    DataType *recv_buff_d[NUMBER_NEIGHBORS]; // Device receive buffers for each neighbor

};
typedef struct Halo_STRUCT Halo;

// Stores halo data used for communication between GPUs and associated metadata.
struct Halo_padded_STRUCT{
    local_int_t nx;                 // Local subdomain size in x
    local_int_t ny;                 // Local subdomain size in y
    local_int_t nz;                 // Local subdomain size in z
    local_int_t dimx;               // Halo dimension in x including ghost layers
    local_int_t dimy;               // Halo dimension in y including ghost layers
    local_int_t dimz;               // Halo dimension in z including ghost layers
    DataType *interior;             // Pointer to interior data array
    DataType *x_d;                 // Device pointer to halo data
    Problem *problem;               // Associated problem metadata

    DataType *send_buff_h[NUMBER_NEIGHBORS]; // Host send buffers for each neighbor
    DataType *recv_buff_h[NUMBER_NEIGHBORS]; // Host receive buffers for each neighbor

    DataType *send_buff_d[NUMBER_NEIGHBORS]; // Device send buffers for each neighbor
    DataType *recv_buff_d[NUMBER_NEIGHBORS]; // Device receive buffers for each neighbor

};
typedef struct Halo_STRUCT Halo_padded;

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

// Initialization functions
void InitGPU(Problem *problem);

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem);

// Halo memory management
void InitHaloMemGPU(Halo *halo, Problem *problem);
void InitHaloMemCPU(Halo *halo, Problem *problem);
void InitHalo(Halo *halo, Problem *problem);
void FreeHaloGPU(Halo *halo);
void FreeHaloCPU(Halo *halo);
void FreeHalo(Halo *halo);

// Halo utility operations
void InjectDataToHalo(Halo *halo, DataType *data);
void SetHaloZeroGPU(Halo *halo);
void SetHaloGlobalIndexGPU(Halo *halo, Problem *problem);
void SetHaloQuotientGlobalIndexGPU(Halo *halo, Problem *problem);
void SetHaloRandomGPU(Halo *halo, Problem *problem, int min, int max, int seed);
void PrintHalo(Halo *x_d);
bool IsHaloZero(Halo *x_d);

// Communication and result handling
void SendResult(int rank_recv, Halo *x_d, Problem *problem);
void GatherResult(Halo *x_d, Problem *problem, DataType *result_h);

// Matrix generation and verification
void GenerateStripedPartialMatrix(Problem *problem, DataType *A);
bool VerifyPartialMatrix(DataType *striped_A_local_h, DataType *striped_A_global_h, int num_stripes, Problem *problem);

// Plane, edge, and corner extraction/injection
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