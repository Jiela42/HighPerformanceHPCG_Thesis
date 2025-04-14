#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=AMGXBench
#SBATCH --account=a-g200
#SBATCH --nodelist=ault25
#SBATCH --output BenchmarkAMGXA100.out
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda/12.1.1 
module load cmake/3.21.3
module load gcc
module load openmpi

cd /users/dknecht/AMGX/buildABenchmark/examples

srun ./hpcg_bench