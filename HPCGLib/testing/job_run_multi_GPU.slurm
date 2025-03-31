#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=hpcg_test
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
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
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests