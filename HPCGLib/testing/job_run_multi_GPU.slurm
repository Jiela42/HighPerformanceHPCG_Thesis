#!/usr/local/bin/bash

##Resources
#SBATCH --job-name hpcgTesting
#SBATCH --output result.out
##SBATCH --partition=amdrtx
#SBATCH --time 00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --distribution=cyclic

##Modules
module purge
module load cuda
module load openmpi
module load cmake/3.21.3

##Compile

##Run
srun ./run_multi_GPU