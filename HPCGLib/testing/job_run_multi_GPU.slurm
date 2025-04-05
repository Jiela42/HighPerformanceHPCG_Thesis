#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench_solver
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1

## Run 
srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 2 2 1 512 512 512 CG