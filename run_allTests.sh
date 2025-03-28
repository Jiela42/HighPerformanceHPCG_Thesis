#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=HPCG_tests
#SBATCH --account=a-g200
#SBATCH --output AllTests_Output.out
#SBATCH --time 00:15:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda
module load cmake
module load gcc
module load cray-mpich


cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib

# remove the old build if it exists
# if [ -d "build" ]; then
#     rm -rf build
# fi
# mkdir build
cd build


# # Build the project
# cmake ..
make -j16


# Navigate to the testing directory
cd testing

# Run the tests
./run_AllTests

