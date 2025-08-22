#ifndef NON_BLOCKING_MPI_CUDA_AWARE_HALO_EXCHANGE_CUH
#define NON_BLOCKING_MPI_CUDA_AWARE_HALO_EXCHANGE_CUH

#include "HPCG_versions/striped_multi_GPU.cuh"

template <typename T>
class non_blocking_cuda_aware_mpi_Implementation : public striped_multi_GPU_Implementation<T> {
public:

    non_blocking_cuda_aware_mpi_Implementation(){
        this->comm_type = "non_blocking_cuda_aware_mpi";
    }

    Problem* init_comm(
        int argc, char *argv[],
        int npx, int npy, int npz,
        local_int_t nx, local_int_t ny, local_int_t nz,
        bool initMPI
    ) override {
        return init_comm_non_blocking_cuda_aware_MPI(argc, argv, npx, npy, npz, nx, ny, nz, initMPI);
    }

    void ExchangeHalo(Halo * halo, Problem * problem
    ) override {
        ExchangeHaloNonBlockingCudaAwareMPI(halo, problem);
    }

    void finalize_comm(Problem *problem) override {
        finalize_comm_non_blocking_cuda_aware_MPI(problem);
    }

    void initialize_COO_comm(
        Problem *problem, 
        striped_partial_Matrix<DataType> &A_local);

private:

    Problem* init_comm_non_blocking_cuda_aware_MPI(
        int argc, char *argv[],
        int npx, int npy, int npz,
        local_int_t nx, local_int_t ny, local_int_t nz,
        bool initMPI
    );

    void ExchangeHaloNonBlockingCudaAwareMPI(
        Halo * halo,
        Problem * problem
    );

    void finalize_comm_non_blocking_cuda_aware_MPI(
        Problem *problem
    );
};

#endif // NON_BLOCKING_CUDA_AWARE_MPI_HALO_EXCHANGE_CUH