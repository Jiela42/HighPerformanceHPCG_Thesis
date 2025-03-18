#ifndef BLOCKING_MPI_HALO_EXCHANGE_CUH
#define BLOCKING_MPI_HALO_EXCHANGE_CUH

#include "HPCG_versions/striped_multi_GPU.cuh"

template <typename T>
class blocking_mpi_Implementation : public striped_multi_GPU_Implementation<T> {
public:

    void ExchangeHalo(Halo * halo, Problem * problem
    ) override {
        ExchangeHaloBlockingMPI(halo, problem);
    }

private:

    void ExchangeHaloBlockingMPI(
        Halo * halo,
        Problem * problem
    );
};

#endif // BLOCKING_MPI_HALO_EXCHANGE_CUH