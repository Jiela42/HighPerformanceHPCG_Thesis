#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench_solver
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 12:00:00
#SBATCH --partition=large
#SBATCH --nodes=512
#SBATCH --ntasks=2048
#SBATCH --gpus-per-task=1

## Run 
for size in 32 64 128 256 512; do
    for npx in 1 2 3 4 5 6 7 8 9 10 11 12; do
        npy=$npx
        npz=$npx
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: 32 | Tasks: 125 | Platform: CLARIDEN GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n $(( npx * npx * npx )) /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size DOT HALO WAXPBY

        echo ""
    done
done

for size in 32 64 128 256 512; do
    for npx in 1 2 3 4 5 6 7 8 9 10 11 12; do
        npy=$npx
        npz=$npx
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: 32 | Tasks: 125 | Platform: CLARIDEN GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n $(( npx * npx * npx )) /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size SPMV

        echo ""
    done
done

for size in 32 64 128 256 512; do
    for npx in 1 2 3 4 5 6 7 8 9 10 11 12; do
        npy=$npx
        npz=$npx
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: 32 | Tasks: 125 | Platform: CLARIDEN GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n $(( npx * npx * npx )) /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size SYMGS

        echo ""
    done
done

for size in 32 64 128 256 512; do
    for npx in 1 2 3 4 5 6 7 8 9 10 11 12; do
        npy=$npx
        npz=$npx
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: 32 | Tasks: 125 | Platform: CLARIDEN GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n $(( npx * npx * npx )) /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size MG CG

        echo ""
    done
done