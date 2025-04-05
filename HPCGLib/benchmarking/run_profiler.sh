#!/bin/bash
## SBATCH --partition=amdrtx
## SBATCH --nodelist=ault41
## SBATCH --time=2:00:00
## SBATCH --job-name=HPCG_Profiler
## SBATCH --output=profiler_output.txt

# Navigate to the benchmarking directory
cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib

# remove the old build
# rm -rf build

# load the necessary modules
module load cuda
module load cmake/3.21.3
module load openmpi

# Build the project
# mkdir build
cd build
# cmake ..
make -j16

# Run the benchmark
cd benchmarking
# srun nsys profile --capture-range=cudaProfilerApi --trace=cuda -o profiler_output --export=json ./run_profiler 32 32 32 "CG" "striped_box_coloring_Implementation"
# srun ncu --profile-from-start off --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed --export csv -o profiler_output ./run_profiler 32 32 32 "CG" "striped_box_coloring_Implementation"

OUTPUT_FOLDER="/users/dknecht/HighPerformanceHPCG_Thesis/profiling_results"

start_time=$(date +%s)
# ncu --query-metrics
# flop_count_dp,
srun ncu --profile-from-start off --quiet --metric dram__write_bytes.sum,gpu__time_duration.sum,dram__utilization.avg,dram__write_throughput.avg,dram__read_bytes.sum,dram__read_throughput.avg -f --export "$OUTPUT_FOLDER/profiler_output.ncu-rep" ./run_profiler 32 32 32 "CG" "striped_box_coloring_Implementation"
ncu --import "$OUTPUT_FOLDER/profiler_output.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_32x32x32_CG_striped_box_coloring_333.csv"
end_time=$(date +%s)
# Calculate elapsed time
execution_time=$((end_time - start_time))
minutes=$((execution_time / 60))
seconds=$((execution_time % 60))

# Print execution time in minutes and seconds
echo "Execution time: $minutes minutes and $seconds seconds"
