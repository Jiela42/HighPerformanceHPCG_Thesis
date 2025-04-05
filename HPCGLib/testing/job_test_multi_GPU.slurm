#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench_solver
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 00:15:00
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1

## Set NCCL debug mode
export NCCL_DEBUG=WARN

## Run 
echo "Started: $(date '+%H:%M:%S')"
srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests
echo "Finished: $(date '+%H:%M:%S')"
