#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=hpcg_test
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 01:00:00
#SBATCH --partition=large
#SBATCH --nodes=512
#SBATCH --ntasks=2048
##SBATCH --gpus-per-task=1

##Modules
#module purge
#module load cuda
#module load cray-mpich
#module load cmake
#module load nccl
#module load gcc

##Compile
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build
##cmake .. && make -j8
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/testing
## cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing

## Run 
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 16 16 16 16 16 16 ALL
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 16 16 16 32 32 32 ALL
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 16 16 16 64 64 64 ALL
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 16 16 16 128 128 128 ALL
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 16 16 16 256 256 256 ALL
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 16 16 16 512 512 512 ALL
#srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests
