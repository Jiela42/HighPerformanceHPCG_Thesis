#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=HBDIA
#SBATCH --account=a-g200
#SBATCH --output HBDIA_result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

##setup
export MPICH_GPU_SUPPORT_ENABLED=1

## Run 
echo "Started: $(date '+%H:%M:%S')"
echo ""

echo "+---------------------------------+"
echo "| Running run_HBDIA_test          |"
echo "+---------------------------------+"
echo ""
srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_HBDIA_test

echo ""
echo "Finished: $(date '+%H:%M:%S')"