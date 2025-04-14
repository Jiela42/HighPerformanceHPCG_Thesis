#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=A100StriperBenchmark
#SBATCH --account=a-g200
#SBATCH --nodelist=ault25
#SBATCH --output StriperBenchmarkA100CORBox.out
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda/12.1.1 

module load cmake/3.21.3
module load gcc
module load openmpi


# Striper Benchmark
cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib/buildABenchmark/benchmarking

srun ./run_full_bench