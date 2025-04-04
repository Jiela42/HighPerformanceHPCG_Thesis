#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=getData
#SBATCH --account=a-g34
#SBATCH --output Data.out
#SBATCH --time 00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda


nvidia-smi
