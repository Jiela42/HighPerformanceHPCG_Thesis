#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=CHOOSE YOUR OWN
#SBATCH --account=a-g34
#SBATCH --output result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

##Modules
module purge
module load cuda
module load cray-mpich
module load cmake
module load nccl

##Compile
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build
##cmake .. && make -j8
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/testing
## cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing

## Run 
srun /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_AllTests