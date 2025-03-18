#!/usr/local/bin/bash

##Resources
#SBATCH --job-name hpcgTesting
#SBATCH --account=g34
#SBATCH --output result.out
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --distribution=block

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

# Run with npx npy npz nx ny nz
srun /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU 3 3 3 8 8 8

# Run with npx npy npz nx ny nz
srun /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU 3 3 3 16 16 16

# Run with npx npy npz nx ny nz
srun /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU 3 3 3 32 32 32