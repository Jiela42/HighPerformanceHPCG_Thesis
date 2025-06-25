#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=hpcg_test_
#SBATCH --account=a-g200
#SBATCH --output /users/nrottstegge/github/HighPerformanceHPCG_Thesis/hpcg/terminal.out
#SBATCH --time 00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18

##setup
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=18
cd /users/nrottstegge/github/HighPerformanceHPCG_Thesis/hpcg

## Run 
echo "Started: $(date '+%H:%M:%S')"

echo "Starting HPCG Benchmark"
srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/hpcg/xhpcg --rt=120 --nx=256

echo "Finished: $(date '+%H:%M:%S')"

echo ""
echo ""

echo "DONE."