// #ifndef NCCL_HALO_EXCHANGE_CUH
// #define NCCL_HALO_EXCHANGE_CUH

// #include "HPCG_versions/striped_multi_GPU.cuh"
// #include <nccl.h>

// template <typename T>
// class NCCL_Implementation : public striped_multi_GPU_Implementation<T> {
// public:

//     Problem* init_comm(
//         int argc, char *argv[],
//         int npx, int npy, int npz,
//         local_int_t nx, local_int_t ny, local_int_t nz
//     ) override {
//         return init_comm_NCCL(argc, argv, npx, npy, npz, nx, ny, nz);
//     }

//     void ExchangeHalo(Halo * halo, Problem * problem
//     ) override {
//         ExchangeHaloNCCL(halo, problem);
//     }

//     void finalize_comm(Problem *problem) override {
//         finalize_comm_NCCL(problem);
//     }

// private:

//     cudaStream_t cuda_stream;
//     ncclComm_t nccl_comm;

//     Problem* init_comm_NCCL(
//         int argc, char *argv[],
//         int npx, int npy, int npz,
//         local_int_t nx, local_int_t ny, local_int_t nz
//     );

//     void ExchangeHaloNCCL(
//         Halo * halo,
//         Problem * problem
//     );

//     void finalize_comm_NCCL(
//         Problem *problem
//     );
// };

// #endif // NCCL_HALO_EXCHANGE_CUH