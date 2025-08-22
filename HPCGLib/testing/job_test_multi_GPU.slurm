#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=bench_solver
#SBATCH --account=a-g200
#SBATCH --output result.out
#SBATCH --time 00:05:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1

##setup
export MPICH_GPU_SUPPORT_ENABLED=1

## Run 
echo "Started: $(date '+%H:%M:%S')"
echo "Testing BLOCKING MPI implementation"
#srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/vadd
#srun ncu --set full --import-source=yes -f --export my_report.ncu-rep /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/vadd
srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU_tests NON_BLOCKING_CUDA_MPI
#srun ncu --set full --call-stack --import-source=yes -f --export my_report.ncu-rep /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU_tests NON_BLOCKING_CUDA_MPI
#srun nsys profile --force-overwrite true -o nsys_report /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU_tests NON_BLOCKING_CUDA_MPI
echo "Finished: $(date '+%H:%M:%S')"

#echo ""
#echo ""

#echo "Started: $(date '+%H:%M:%S')"
#echo "Testing NON_BLOCKING HOST_ONLY MPI implementation"
#srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU_tests NON_BLOCKING_HOST_MPI
#echo "Finished: $(date '+%H:%M:%S')"

#echo ""
#echo ""

#echo "Started: $(date '+%H:%M:%S')"
#echo "Testing NON_BLOCKING CUDA_AWARE MPI implementation"
#srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU_tests NON_BLOCKING_CUDA_MPI
#echo "Finished: $(date '+%H:%M:%S')"

#echo ""
#echo ""

#echo "Started: $(date '+%H:%M:%S')"
#echo "Testing NCCL implementation"
#srun /users/nrottstegge/github/HighPerformanceHPCG_Thesis/build/HPCGLib/testing/run_multi_GPU_tests NCCL
#echo "Finished: $(date '+%H:%M:%S')"

#echo ""
#echo ""

echo "DONE."