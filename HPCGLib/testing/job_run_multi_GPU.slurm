#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench_solver
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
##SBATCH --gpus-per-task=1

#Modules
module purge
module load cuda
module load cray-mpich
module load cmake
module load nccl
module load gcc

#Compile
cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib/build
make -j8

##cmake .. && make -j8
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/testing
## cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing

## Run 
#srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 2 2 1 512 512 512 DOT
# srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests
srun /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests
