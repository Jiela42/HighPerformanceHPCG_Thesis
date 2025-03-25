#!/usr/local/bin/bash

##Resources
#SBATCH --job-name hpcgTesting
#SBATCH --output result.out
#SBATCH --time 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug

##Modules
#module purge
#module load cuda
#module load openmpi
#module load cmake/3.21.3

##Compile
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build
##cmake .. && make -j8
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/testing
cd /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/testing

##Run
srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests