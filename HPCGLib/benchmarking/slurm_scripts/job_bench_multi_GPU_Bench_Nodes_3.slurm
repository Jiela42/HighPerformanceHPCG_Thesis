#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench2
#SBATCH --account=g34
#SBATCH --output=/users/ybudanaz/nils/HighPerformanceHPCG_Thesis/dummy_timing_results/result_2.out
#SBATCH --time 00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --gpus-per-task=1

## Define number of nodes as a variable
NODES=3
GPUS_PER_NODE=4

## Run 
declare -a configs=(
  "1 1 1" #1
  "2 1 1" #2
  "2 2 1" #4
  "2 2 2" #8
  "4 2 2" #16
  "4 4 2" #32
  "4 4 4" #64
  "8 4 4" #128
  "8 8 4" #256
  "8 8 8" #512
  "16 8 8" #1024
  "16 16 8" #2048
)

for size in 32 64 128 256 512; do
    for config in "${configs[@]}"; do
        read -r npx npy npz <<< "$config"
        TASKS=$((npx * npy * npz))
        REQUIRED_NODES=$(( (TASKS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
        if (( REQUIRED_NODES > NODES )); then
            echo "Skipping TASKS=${TASKS} (requires ${REQUIRED_NODES} nodes, only ${NODES} available)"
            continue
        fi
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: ${NODES} | Tasks: ${TASKS} | CLUSTER: DAINT | GPU: GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n ${TASKS} /users/ybudanaz/nils/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size DOT HALO WAXPBY

        echo ""
    done
done

for size in 32 64 128 256 512; do
    for config in "${configs[@]}"; do
        read -r npx npy npz <<< "$config"
        TASKS=$((npx * npy * npz))
        REQUIRED_NODES=$(( (TASKS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
        if (( REQUIRED_NODES > NODES )); then
            echo "Skipping TASKS=${TASKS} (requires ${REQUIRED_NODES} nodes, only ${NODES} available)"
            continue
        fi
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: ${NODES} | Tasks: ${TASKS} | CLUSTER: CLARIDEN | GPU: GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n ${TASKS} /users/ybudanaz/nils/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size SPMV

        echo ""
    done
done

for size in 32 64 128 256 512; do
    for config in "${configs[@]}"; do
        read -r npx npy npz <<< "$config"
        TASKS=$((npx * npy * npz))
        REQUIRED_NODES=$(( (TASKS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
        if (( REQUIRED_NODES > NODES )); then
            echo "Skipping TASKS=${TASKS} (requires ${REQUIRED_NODES} nodes, only ${NODES} available)"
            continue
        fi
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: ${NODES} | Tasks: ${TASKS} | CLUSTER: CLARIDEN | GPU: GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n ${TASKS} /users/ybudanaz/nils/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size MG

        echo ""
    done
done

for size in 32 64 128 256 512; do
    for config in "${configs[@]}"; do
        read -r npx npy npz <<< "$config"
        TASKS=$((npx * npy * npz))
        REQUIRED_NODES=$(( (TASKS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
        if (( REQUIRED_NODES > NODES )); then
            echo "Skipping TASKS=${TASKS} (requires ${REQUIRED_NODES} nodes, only ${NODES} available)"
            continue
        fi
        echo "************************************************************"
        echo "* Running benchmark: NX=${size} NY=${size} NZ=${size}"
        echo "* Grid: NPX=${npx} NPY=${npy} NPZ=${npz}"
        echo "* Total Nodes: ${NODES} | Tasks: ${TASKS} | CLUSTER: CLARIDEN | GPU: GH200"
        echo "* Currently Loaded Modules: gcc/13.3.0   cmake/3.30.5   cuda/12.6.2   cray-mpich/8.1.30   nccl/2.22.3-1"
        echo "************************************************************"
        
        srun -n ${TASKS} /users/ybudanaz/nils/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU $npx $npy $npz $size $size $size CG

        echo ""
    done
done