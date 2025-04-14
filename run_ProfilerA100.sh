#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=A100Profiler
#SBATCH --account=a-g200
#SBATCH --nodelist=ault25
#SBATCH --output ProfilerAMGX128A100.out
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda/12.1.1 
module load cmake/3.21.3
module load gcc
module load openmpi

# # Striper 444 NCU
# echo "Starting profiling for Striped Box coloring (coloringBox 4x4x4) on A100"
# cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib/buildABenchmark/benchmarking

# OUTPUT_FOLDER="/users/dknecht/HighPerformanceHPCG_Thesis/profiling_results"
# NX_LIST=(32 64 128 256)
# MACHINE="A100"

# start_time=$(date +%s)

# # Iterate through the list of NX values
# for NX in "${NX_LIST[@]}"; do
#     METHOD="SymGS"
#     IMPLEMENTATION="Striped Box coloring (coloringBox 4x4x4)"
#     echo "Starting profiling for ${METHOD} with striped_box_coloring_Implementation for size ${NX}x${NX}x${NX} on ${MACHINE}"

#     # Profiling for the specified size
#     step_start_time=$(date +%s)
#     srun ncu --profile-from-start off \
#     --set full \
#     -f --export "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" \
#     ./run_profiler $NX $NX $NX $METHOD "striped_box_coloring_Implementation"

#     if [ $? -ne 0 ]; then
#         echo "Error: Profiling failed for ${NX}x${NX}x${NX} on ${MACHINE}"
#         continue
#     fi

#     echo "Profiler output for ${MACHINE}_${NX}x${NX}x${NX} generated at $OUTPUT_FOLDER/profiler_output_${MACHINE}_${METHOD}_${NX}x${NX}x${NX}.ncu-rep"

#     # Export the profiling results to CSV
#     ncu --import "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_444.csv"

#     METADATA="${IMPLEMENTATION},${MACHINE},${NX}x${NX}x${NX},${METHOD}"
#     sed -i "1i $METADATA" "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_444.csv"

#     step_end_time=$(date +%s)
#     step_execution_time=$((step_end_time - step_start_time))
#     echo "Execution time for ${MACHINE}_${NX}x${NX}x${NX}: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"
# done

# end_time=$(date +%s)

# # Calculate total elapsed time
# execution_time=$((end_time - start_time))
# minutes=$((execution_time / 60))
# seconds=$((execution_time % 60))

# echo "Total execution time: $minutes minutes and $seconds seconds"

# # Striper 333 NCU
# echo "Starting profiling for Striped Box coloring (coloringBox 3x3x3) on A100"
# cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib/buildA333/benchmarking

# start_time=$(date +%s)

# # Iterate through the list of NX values
# for NX in "${NX_LIST[@]}"; do
#     METHOD="SymGS"
#     IMPLEMENTATION="Striped Box coloring (coloringBox 3x3x3)"
#     echo "Starting profiling for ${METHOD} with striped_box_coloring_Implementation for size ${NX}x${NX}x${NX} on ${MACHINE}"

#     # Profiling for the specified size
#     step_start_time=$(date +%s)
#     srun ncu --profile-from-start off \
#     --set full \
#     -f --export "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" \
#     ./run_profiler $NX $NX $NX $METHOD "striped_box_coloring_Implementation"

#     if [ $? -ne 0 ]; then
#         echo "Error: Profiling failed for ${NX}x${NX}x${NX} on ${MACHINE}"
#         continue
#     fi

#     echo "Profiler output for ${MACHINE}_${NX}x${NX}x${NX} generated at $OUTPUT_FOLDER/profiler_output_${MACHINE}_${METHOD}_${NX}x${NX}x${NX}.ncu-rep"

#     # Export the profiling results to CSV
#     ncu --import "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_333.csv"

#     METADATA="${IMPLEMENTATION},${MACHINE},${NX}x${NX}x${NX},${METHOD}"
#     sed -i "1i $METADATA" "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_333.csv"

#     step_end_time=$(date +%s)
#     step_execution_time=$((step_end_time - step_start_time))
#     echo "Execution time for ${MACHINE}_${NX}x${NX}x${NX}: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"
# done

# end_time=$(date +%s)

# # Calculate total elapsed time
# execution_time=$((end_time - start_time))
# minutes=$((execution_time / 60))
# seconds=$((execution_time % 60))

# echo "Total execution time: $minutes minutes and $seconds seconds"


# # Striper 222 NCU
# echo "Starting profiling for Striped Box coloring (coloringBox 2x2x2) on A100"
# cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib/buildA222/benchmarking

# start_time=$(date +%s)

# # Iterate through the list of NX values
# for NX in "${NX_LIST[@]}"; do
#     METHOD="SymGS"
#     IMPLEMENTATION="Striped Box coloring (coloringBox 2x2x2)"
#     echo "Starting profiling for ${METHOD} with striped_box_coloring_Implementation for size ${NX}x${NX}x${NX} on ${MACHINE}"

#     # Profiling for the specified size
#     step_start_time=$(date +%s)
#     srun ncu --profile-from-start off \
#     --set full \
#     -f --export "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" \
#     ./run_profiler $NX $NX $NX $METHOD "striped_box_coloring_Implementation"

#     if [ $? -ne 0 ]; then
#         echo "Error: Profiling failed for ${NX}x${NX}x${NX} on ${MACHINE}"
#         continue
#     fi

#     echo "Profiler output for ${MACHINE}_${NX}x${NX}x${NX} generated at $OUTPUT_FOLDER/profiler_output_${MACHINE}_${METHOD}_${NX}x${NX}x${NX}.ncu-rep"

#     # Export the profiling results to CSV
#     ncu --import "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_222.csv"

#     METADATA="${IMPLEMENTATION},${MACHINE},${NX}x${NX}x${NX},${METHOD}"
#     sed -i "1i $METADATA" "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_222.csv"

#     step_end_time=$(date +%s)
#     step_execution_time=$((step_end_time - step_start_time))
#     echo "Execution time for ${MACHINE}_${NX}x${NX}x${NX}: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"
# done

# end_time=$(date +%s)

# # Calculate total elapsed time
# execution_time=$((end_time - start_time))
# minutes=$((execution_time / 60))
# seconds=$((execution_time % 60))

# echo "Total execution time: $minutes minutes and $seconds seconds"

# AMGX NCU
echo "Starting AMGX profiling on A100"

cd /users/dknecht/AMGX/buildAProfiling/examples

OUTPUT_FOLDER="/users/dknecht/HighPerformanceHPCG_Thesis/profiling_results/AMGX"
NX_LIST=(128)

start_time=$(date +%s)

# Iterate through the list of NX values
for NX in "${NX_LIST[@]}"; do
    METHOD="SymGS"
    echo "Starting AMGX profiling for ${METHOD} for size ${NX}x${NX}x${NX} on ${MACHINE}"

    # Profiling for the specified size
    step_start_time=$(date +%s)
    srun ncu --profile-from-start off \
    --set full \
    -f --export "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" \
    ./hpcg_bench $NX $NX $NX $METHOD

    if [ $? -ne 0 ]; then
        echo "Error: Profiling failed for ${NX}x${NX}x${NX} on ${MACHINE}"
        continue
    fi

    echo "Profiler output for ${MACHINE}_${NX}x${NX}x${NX} generated at $OUTPUT_FOLDER/profiler_output_${MACHINE}_${METHOD}_${NX}x${NX}x${NX}.ncu-rep"

    # Export the profiling results to CSV
    ncu --import "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_AMGX.csv"

    METADATA="AMGX,${MACHINE},${NX}x${NX}x${NX},${METHOD}"
    sed -i "1i $METADATA" "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_AMGX.csv"

    step_end_time=$(date +%s)
    step_execution_time=$((step_end_time - step_start_time))
    echo "Execution time for ${MACHINE}_${NX}x${NX}x${NX}: $((step_execution_time / 60)) minutes and $((step_execution_time % 60)) seconds"
done

end_time=$(date +%s)

# Calculate total elapsed time
execution_time=$((end_time - start_time))
minutes=$((execution_time / 60))
seconds=$((execution_time % 60))

# Print total execution time
echo "Total execution time: $minutes minutes and $seconds seconds"


