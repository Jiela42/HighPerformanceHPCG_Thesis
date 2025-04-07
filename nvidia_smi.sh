#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=getData
#SBATCH --account=a-g200
#SBATCH --output Data.out
#SBATCH --time 00:10:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1


module purge

module load cuda

# Check for NVIDIA profilers
srun nvidia-smi

srun nvidia-smi --query-gpu=l2_cache_size --format=csv,noheader,nounits

echo "done"