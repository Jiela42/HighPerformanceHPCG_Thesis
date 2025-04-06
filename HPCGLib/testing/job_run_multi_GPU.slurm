#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=hpcg_test
#SBATCH --output result.out
#SBATCH --time 00:20:00
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1

##Modules
#module purge
#module load cuda
#module load cray-mpich
#module load cmake
#module load nccl
#module load gcc

spack load hip@6.2.1
spack load cmake@3.30.5
spack load hipsparse@6.2.1
spack load hipblas@6.2.1
spack load rocm-cmake@6.2.1
spack load rocm-core@6.2.1
spack load rocsparse@6.2.1
spack load rocblas@6.2.1
spack load rocthrust@6.2.1
spack load rccl@6.2.1
spack load mpich@4.2.3/dfsmhyo
spack load rocprim@6.2.1

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_CXX=hipcc

##Compile
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build
##cmake .. && make -j8
##cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/testing
## cd /users/nrottste/HighPerformanceHPCG_Thesis/HPCGLib/build/testing

## Run
#srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/benchmarking/run_bench_multi_GPU 2 2 1 512 512 512 DOT
#srun /users/ljingjin/HPCG/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests

mpirun -n 4 /capstor/scratch/cscs/ybudanaz/jingjing/HighPerformanceHPCG_Thesis/HPCGLib/build/testing/run_multi_GPU_tests
#mpirun -n 4 xterm -e gdb -x gdb_cmds
#mpirun -n 4 gdb -batch -x gdb_cmds