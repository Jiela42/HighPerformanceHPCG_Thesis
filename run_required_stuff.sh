#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=HPCG_required_runs
#SBATCH --account=a-g200
#SBATCH --output necessary_runs.out
#SBATCH --time 8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda
module load cmake
module load gcc
module load cray-mpich


# we build amgx
cd /users/dknecht/amgx_expanded_for_HPCG

# remove the old build if it exists
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build

ls
cd build

# Build the project
cmake ..
make -j16


cd examples

# we run ncu on amgx

OUTPUT_FOLDER="/users/dknecht/HighPerformanceHPCG_Thesis/profiling_results/AMGX"
NX_LIST=(32 64 128 256 512)
MACHINE="GH200"


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

# we build our stuff
cd /users/dknecht/HighPerformanceHPCG_Thesis/HPCGLib

# remove the old build if it exists
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build

cd build
# # Build the project
cmake ..
make -j16

# we run full bench
cd benchmarking

# we run ncu on 2x2x2 our stuff
OUTPUT_FOLDER="/users/dknecht/HighPerformanceHPCG_Thesis/profiling_results"
NX_LIST=(32 64 128 256 512)
MACHINE="GH200"

start_time=$(date +%s)

# Iterate through the list of NX values
for NX in "${NX_LIST[@]}"; do
    METHOD="SymGS"
    IMPLEMENTATION="Striped Box coloring (coloringBox 2x2x2)"
    echo "Starting profiling for ${METHOD} with striped_box_coloring_Implementation for size ${NX}x${NX}x${NX} on ${MACHINE}"

    # Profiling for the specified size
    step_start_time=$(date +%s)
    srun ncu --profile-from-start off \
    --set full \
    -f --export "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" \
    ./run_profiler $NX $NX $NX $METHOD "striped_box_coloring_Implementation"

    if [ $? -ne 0 ]; then
        echo "Error: Profiling failed for ${NX}x${NX}x${NX} on ${MACHINE}"
        continue
    fi

    echo "Profiler output for ${MACHINE}_${NX}x${NX}x${NX} generated at $OUTPUT_FOLDER/profiler_output_${MACHINE}_${METHOD}_${NX}x${NX}x${NX}.ncu-rep"

    # Export the profiling results to CSV
    ncu --import "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}.ncu-rep" --csv > "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_222.csv"

    METADATA="${IMPLEMENTATION},${MACHINE},${NX}x${NX}x${NX},${METHOD}"
    sed -i "1i $METADATA" "$OUTPUT_FOLDER/profiler_output_${MACHINE}_${NX}x${NX}x${NX}_${METHOD}_striped_box_coloring_222.csv"

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

srun ./run_full_bench