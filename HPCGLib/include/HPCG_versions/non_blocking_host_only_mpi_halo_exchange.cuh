#ifndef NON_BLOCKING_MPI_HOST_ONLY_HALO_EXCHANGE_CUH
#define NON_BLOCKING_MPI_HOST_ONLY_HALO_EXCHANGE_CUH

#include "HPCG_versions/striped_multi_GPU.cuh"

template <typename T>
class non_blocking_host_only_mpi_Implementation : public striped_multi_GPU_Implementation<T> {
public:

    non_blocking_host_only_mpi_Implementation(){
        this->comm_type = "non_blocking_host_only_mpi";
    }

    Problem* init_comm(
        int argc, char *argv[],
        int npx, int npy, int npz,
        local_int_t nx, local_int_t ny, local_int_t nz,
        bool initMPI
    ) override {
        return init_comm_non_blocking_host_only_MPI(argc, argv, npx, npy, npz, nx, ny, nz, initMPI);
    }

    void ExchangeHalo(Halo * halo, Problem * problem
    ) override {
        ExchangeHaloNonBlockingHostOnlyMPI(halo, problem);
    }

    void finalize_comm(Problem *problem) override {
        finalize_comm_non_blocking_host_only_MPI(problem);
    }

private:

    Problem* init_comm_non_blocking_host_only_MPI(
        int argc, char *argv[],
        int npx, int npy, int npz,
        local_int_t nx, local_int_t ny, local_int_t nz,
        bool initMPI
    );

    void ExchangeHaloNonBlockingHostOnlyMPI(
        Halo * halo,
        Problem * problem
    );

    void finalize_comm_non_blocking_host_only_MPI(
        Problem *problem
    );
};

#endif // NON_BLOCKING_HOST_ONLY_MPI_HALO_EXCHANGE_CUH