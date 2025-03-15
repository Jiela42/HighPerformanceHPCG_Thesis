#!/usr/local/bin/bash

##Resources
#SBATCH --job-name hpcgTesting
#SBATCH --output result.out
##SBATCH --partition=amdrtx
#SBATCH --time 00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=27
#SBATCH --gres=gpu:v100:4
#SBATCH --distribution=cyclic
##SBATCH --partition=amdv100

##Modules
module purge
module load cuda
module load openmpi
module load cmake/3.21.3

##Compile
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build
##cmake .. && make -j8
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/testing
cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing

##Run
srun /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU