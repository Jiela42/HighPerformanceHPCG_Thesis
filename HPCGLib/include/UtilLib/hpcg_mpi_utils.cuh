#ifndef HPCG_MPI_UTILS_CUH
#define HPCG_MPI_UTILS_CUH

typedef int local_int_t;
typedef int global_int_t;

using DataType = double;

#define MPIDataType MPI_DOUBLE

struct Problem_STRUCT{
    int npx, npy, npz;
    local_int_t nx, ny, nz;
    int size, rank;
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

void GenerateProblem(int npx, int npy, int npz, local_int_t nx, local_int_t ny, local_int_t nz, int size, int rank, Problem *problem);
void InitHaloMemGPU(Halo *halo, int nx, int ny, int nz);
void SetHaloZeroGPU(Halo *halo);
void FreeHaloGPU(Halo *halo);
void InitGPU(Problem *problem);
void ExchangeHalo(Halo *halo, Problem *problem);

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