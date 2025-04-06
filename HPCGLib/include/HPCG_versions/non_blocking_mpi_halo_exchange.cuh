#ifndef NON_BLOCKING_MPI_HALO_EXCHANGE_CUH
#define NON_BLOCKING_MPI_HALO_EXCHANGE_CUH

#include "HPCG_versions/striped_multi_GPU.cuh"

template <typename T>
class non_blocking_mpi_Implementation : public striped_multi_GPU_Implementation<T> {
public:

Problem* init_comm(
    int argc, char *argv[],
    int npx, int npy, int npz,
    local_int_t nx, local_int_t ny, local_int_t nz
) override {
    return init_comm_non_blocking_MPI(argc, argv, npx, npy, npz, nx, ny, nz);
}

void ExchangeHalo(Halo * halo, Problem * problem
) override {
    ExchangeHaloNonBlockingMPI(halo, problem);
}

void finalize_comm(Problem *problem) override {
    finalize_comm_non_blocking_MPI(problem);
}

private:

    Problem* init_comm_non_blocking_MPI(
        int argc, char *argv[],
        int npx, int npy, int npz,
        local_int_t nx, local_int_t ny, local_int_t nz
    );

    void ExchangeHaloNonBlockingMPI(
        Halo * halo,
        Problem * problem
    );

    void finalize_comm_non_blocking_MPI(
        Problem *problem
    );
};

#endif // NON_BLOCKING_MPI_HALO_EXCHANGE_CUH