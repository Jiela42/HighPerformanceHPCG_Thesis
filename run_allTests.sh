#!/bin/bash
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault43
#SBATCH --time=4:00:00
#SBATCH --job-name=HPCG_SingleNode_Test
#SBATCH --output=AllTests_Output.txt

module load cuda/12.1.1
module load cmake/3.21.3
module load openmpi


cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib

# remove the old build if it exists
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build


# Build the project
cmake ..
make -j16


# Navigate to the testing directory
cd testing

# Run the tests
./run_AllTests


#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault43
#SBATCH --time=4:00:00
#SBATCH --job-name=HPCG_SingleNode_Test
#SBATCH --output=AllTests_Output.txt
#SBATCH --open-mode=append

module purge
module load cuda/12.1.1
module load cmake/3.21.3
module load openmpi

cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib

# remove the old build
rm -rf build
mkdir build
cd build

# Build the project
cmake ..
make -j16

# Navigate to the testing directory
cd testing

# Run the tests
./run_AllTests
