#ifndef NON_BLOCKING_MPI_HALO_EXCHANGE_CUH
#define NON_BLOCKING_MPI_HALO_EXCHANGE_CUH

#include "HPCG_versions/striped_multi_GPU.cuh"

template <typename T>
class non_blocking_mpi_Implementation : public striped_multi_GPU_Implementation<T> {
public:

    void ExchangeHalo(Halo * halo, Problem * problem
    ) override {
        ExchangeHaloNonBlockingMPI(halo, problem);
    }

private:

    void ExchangeHaloNonBlockingMPI(
        Halo * halo,
        Problem * problem
    );
};

#endif // NON_BLOCKING_MPI_HALO_EXCHANGE_CUH