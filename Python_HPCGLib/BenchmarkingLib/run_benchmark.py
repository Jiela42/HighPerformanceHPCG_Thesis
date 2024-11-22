import sys
import os
import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.baseTorch_benchmark import run_BaseTorch_benchmark

# make new timestamped folder in data to avoid overwriting old data
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = "../../timing_results/"

new_folder_path = os.path.join(base_path, timestamp)

overall_start = time.time()

print("****************************************RUNNING BENCHMARK SUITE****************************************", flush=True)

run_BaseTorch_benchmark(8, 8, 8, new_folder_path)
print(f"BaseTorch Finished 8x8x8", flush=True)
run_BaseTorch_benchmark(16, 16, 16, new_folder_path)
print(f"BaseTorch Finished 16x16x16", flush=True)
run_BaseTorch_benchmark(24, 24, 24, new_folder_path)
print(f"BaseTorch Finished 24x24x24", flush=True)
run_BaseTorch_benchmark(32, 32, 32, new_folder_path)
print(f"BaseTorch Finished 32x32x32", flush=True)
run_BaseTorch_benchmark(64, 64, 64, new_folder_path)
print(f"BaseTorch Finished 64x64x64", flush=True)
run_BaseTorch_benchmark(128, 128, 128, new_folder_path)


overall_end = time.time()

time_elapsed = overall_end - overall_start
minutes, seconds = divmod(time_elapsed, 60)

print("Timing finished")
print(f"Timing took: {time_elapsed}")

print("**************************************************************************************************", flush=True)
print("****************************************ALL BENCHMARKS DONE***************************************", flush=True)
print("**************************************************************************************************", flush=True)