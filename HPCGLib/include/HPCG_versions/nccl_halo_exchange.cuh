#ifndef NCCL_HALO_EXCHANGE_CUH
#define NCCL_HALO_EXCHANGE_CUH

#include "HPCG_versions/striped_multi_GPU.cuh"

template <typename T>
class NCCL_Implementation : public striped_multi_GPU_Implementation<T> {
public:

    void ExchangeHalo(Halo * halo, Problem * problem
    ) override {
        ExchangeHaloNCCL(halo, problem);
    }

private:

    void ExchangeHaloNCCL(
        Halo * halo,
        Problem * problem
    );
};

#endif // NCCL_HALO_EXCHANGE_CUH