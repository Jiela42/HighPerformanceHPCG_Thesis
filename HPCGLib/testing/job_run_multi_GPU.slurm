#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench_solver
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1

##setup
export MPICH_GPU_SUPPORT_ENABLED=1

## Run 
echo "Started: $(date '+%H:%M:%S')"
srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU
echo "Finished: $(date '+%H:%M:%S')"

echo "DONE."