#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=getData
#SBATCH --account=a-g34
#SBATCH --output Data.out
#SBATCH --time 00:10:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1


module purge

module load cuda

# Check for NVIDIA profilers
echo "Checking for NVIDIA profilers..."
echo "Searching for nvprof:"
which nvprof || echo "nvprof not found in PATH"

echo "Searching for ncu (Nsight Compute):"
which ncu || echo "ncu not found in PATH"

echo "Searching for nsys (Nsight Systems):"
which nsys || echo "nsys not found in PATH"

# Optionally, search in CUDA directories
echo "Searching in CUDA installation directories..."
find /usr/local/cuda/ -name nvprof 2>/dev/null || echo "nvprof not found in CUDA directories"
find /usr/local/cuda/ -name ncu 2>/dev/null || echo "ncu not found in CUDA directories"
find /usr/local/cuda/ -name nsys 2>/dev/null || echo "nsys not found in CUDA directories"


echo "done"