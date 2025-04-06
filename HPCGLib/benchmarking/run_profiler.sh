#!/bin/bash
##Resources
#SBATCH --job-name=profiling
#SBATCH --account=a-g200
#SBATCH --output profiling.out
#SBATCH --time 01:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda
module load cmake
module load gcc
module load cray-mpich

# Navigate to the benchmarking directory
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

# Run the benchmark
cd benchmarking


OUTPUT_FOLDER="/users/dknecht/HighPerformanceHPCG_Thesis/profiling_results"

start_time=$(date +%s)

sampling_rate=1
# ncu --help

# Profiling for 32x32x32
step_start_time=$(date +%s)
srun ncu --profile-from-start off --quiet \
--metrics dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg,gpu__time_duration.sum,gpu__dram_throughput.avg \
-f --export "$OUTPUT_FOLDER/profiler_output_32x32x32.ncu-rep" \
./run_profiler 32 32 32 "CG" "striped_box_coloring_Implementation"

ncu --import "$OUTPUT_FOLDER/profiler_output_32x32x32.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_32x32x32_CG_striped_box_coloring_333.csv"
step_end_time=$(date +%s)
step_execution_time=$((step_end_time - step_start_time))
echo "Execution time for 32x32x32: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"

# Profiling for 64x64x64
step_start_time=$(date +%s)
srun ncu --profile-from-start off --quiet \
--metrics dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg,gpu__time_duration.sum,gpu__dram_throughput.avg \
-f --export "$OUTPUT_FOLDER/profiler_output_64x64x64.ncu-rep" \
./run_profiler 64 64 64 "CG" "striped_box_coloring_Implementation"

ncu --import "$OUTPUT_FOLDER/profiler_output_64x64x64.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_64x64x64_CG_striped_box_coloring_333.csv"
step_end_time=$(date +%s)
step_execution_time=$((step_end_time - step_start_time))
echo "Execution time for 64x64x64: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"

# Profiling for 128x128x128
step_start_time=$(date +%s)
srun ncu --profile-from-start off --quiet \
--metrics dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg,gpu__time_duration.sum,gpu__dram_throughput.avg \
-f --export "$OUTPUT_FOLDER/profiler_output_128x128x128.ncu-rep" \
./run_profiler 128 128 128 "CG" "striped_box_coloring_Implementation"

ncu --import "$OUTPUT_FOLDER/profiler_output_128x128x128.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_128x128x128_CG_striped_box_coloring_333.csv"
step_end_time=$(date +%s)
step_execution_time=$((step_end_time - step_start_time))
echo "Execution time for 128x128x128: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"

# Profiling for 256x256x256
step_start_time=$(date +%s)
srun ncu --profile-from-start off --quiet \
--metrics dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg,gpu__time_duration.sum,gpu__dram_throughput.avg \
-f --export "$OUTPUT_FOLDER/profiler_output_256x256x256.ncu-rep" \
./run_profiler 256 256 256 "CG" "striped_box_coloring_Implementation"
ncu --import "$OUTPUT_FOLDER/profiler_output_256x256x256.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_256x256x256_CG_striped_box_coloring_333.csv"
step_end_time=$(date +%s)
step_execution_time=$((step_end_time - step_start_time))
echo "Execution time for 256x256x256: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"
# Profiling for 512x512x512
step_start_time=$(date +%s)
srun ncu --profile-from-start off --quiet \
--metrics dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg,gpu__time_duration.sum,gpu__dram_throughput.avg \
-f --export "$OUTPUT_FOLDER/profiler_output_512x512x512.ncu-rep" \
./run_profiler 512 512 512 "CG" "striped_box_coloring_Implementation"
ncu --import "$OUTPUT_FOLDER/profiler_output_512x512x512.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_512x512x512_CG_striped_box_coloring_333.csv"
step_end_time=$(date +%s)
step_execution_time=$((step_end_time - step_start_time))
echo "Execution time for 512x512x512: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"

end_time=$(date +%s)

# Calculate total elapsed time
execution_time=$((end_time - start_time))
minutes=$((execution_time / 60))
seconds=$((execution_time % 60))

# Print total execution time
echo "Total execution time: $minutes minutes and $seconds seconds"
