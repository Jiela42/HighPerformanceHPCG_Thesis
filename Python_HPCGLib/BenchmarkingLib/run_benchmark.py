import sys
import os
import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.baseTorch_benchmark import run_BaseTorch_benchmark
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.matlabReference_benchmark import run_MatlabReference_benchmark
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.BaseCuPy_benchmark import run_BaseCuPy_benchmark
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.NaiveBandedCuPy_benchmark import run_NaiveBandedCuPy_benchmark
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.AMGX_benchmark import run_AMGX_benchmark


# make new timestamped folder in data to avoid overwriting old data
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = "../../timing_results/"

# we use this to overwrite the other path for testing
# base_path = "../test_timing_results/"

new_folder_path = os.path.join(base_path, timestamp)

overall_start = time.time()

print("***********************************RUNNING BENCHMARK SUITE***********************************", flush=True)

# run_BaseTorch_benchmark(8, 8, 8, new_folder_path)
# print(f"BaseTorch Finished 8x8x8", flush=True)
# run_BaseTorch_benchmark(16, 16, 16, new_folder_path)
# print(f"BaseTorch Finished 16x16x16", flush=True)
# run_BaseTorch_benchmark(24, 24, 24, new_folder_path)
# print(f"BaseTorch Finished 24x24x24", flush=True)
# run_BaseTorch_benchmark(32, 32, 32, new_folder_path)
# print(f"BaseTorch Finished 32x32x32", flush=True)
# run_BaseTorch_benchmark(64, 64, 64, new_folder_path)
# print(f"BaseTorch Finished 64x64x64", flush=True)
# run_BaseTorch_benchmark(128, 64, 64, new_folder_path)
# print(f"BaseTorch Finished 128x64x64", flush=True)
# run_BaseTorch_benchmark(128, 128, 128, new_folder_path)
# print(f"BaseTorch Finished 128x128x128", flush=True)
# run_BaseTorch_benchmark(128, 128, 64, new_folder_path)
# run_BaseTorch_benchmark(256, 128, 128, new_folder_path)

# print("BaseTorch benchmarks done", flush=True)
# print("Starting MatlabReference benchmarks", flush=True)
# run_MatlabReference_benchmark(8, 8, 8, new_folder_path)
# print(f"MatlabReference Finished 8x8x8", flush=True)
# run_MatlabReference_benchmark(16, 16, 16, new_folder_path)
# print(f"MatlabReference Finished 16x16x16", flush=True)
# run_MatlabReference_benchmark(24, 24, 24, new_folder_path)
# print(f"MatlabReference Finished 24x24x24", flush=True)
# run_MatlabReference_benchmark(32, 32, 32, new_folder_path)
# print(f"MatlabReference Finished 32x32x32", flush=True)
# run_MatlabReference_benchmark(64, 64, 64, new_folder_path)
# print(f"MatlabReference Finished 64x64x64", flush=True)
# run_MatlabReference_benchmark(128, 128, 128, new_folder_path)
# print(f"MatlabReference Finished 128x128x128", flush=True)


# print("MatlabReference benchmarks done", flush=True)

print("Starting BaseCuPy benchmarks", flush=True)
run_BaseCuPy_benchmark(8, 8, 8, new_folder_path)
run_BaseCuPy_benchmark(16, 16, 16, new_folder_path)
# run_BaseCuPy_benchmark(24, 24, 24, new_folder_path)
print(f"BaseCuPy Finished 24x24x24", flush=True)
run_BaseCuPy_benchmark(32, 32, 32, new_folder_path)
run_BaseCuPy_benchmark(64, 64, 64, new_folder_path)
print(f"BaseCuPy Finished 64x64x64", flush=True)
run_BaseCuPy_benchmark(128, 64, 64, new_folder_path)
print(f"BaseCuPy Finished 128x64x64", flush=True)
run_BaseCuPy_benchmark(128, 128, 64, new_folder_path)
# print(f"BaseCuPy Finished 128x128x64", flush=True)
# run_BaseCuPy_benchmark(128, 128, 128, new_folder_path)
# # the following run out of memory
# # run_BaseCuPy_benchmark(256, 128, 128, new_folder_path)
# # run_BaseCuPy_benchmark(256, 256, 128, new_folder_path)
# print("BaseCuPy benchmarks done", flush=True)

# print("Starting NaiveBandedCuPy benchmarks", flush=True)
# run_NaiveBandedCuPy_benchmark(8, 8, 8, new_folder_path)
# run_NaiveBandedCuPy_benchmark(16, 16, 16, new_folder_path)
# run_NaiveBandedCuPy_benchmark(24, 24, 24, new_folder_path)
# print(f"NaiveBandedCuPy Finished 24x24x24", flush=True)
# run_NaiveBandedCuPy_benchmark(32, 32, 32, new_folder_path)
# run_NaiveBandedCuPy_benchmark(64, 64, 64, new_folder_path)
# print(f"NaiveBandedCuPy Finished 64x64x64", flush=True)
# run_NaiveBandedCuPy_benchmark(128, 64, 64, new_folder_path)
# print(f"NaiveBandedCuPy Finished 128x64x64", flush=True)
# run_NaiveBandedCuPy_benchmark(128, 128, 64, new_folder_path)
# print(f"NaiveBandedCuPy Finished 128x128x64", flush=True)
# run_NaiveBandedCuPy_benchmark(128, 128, 128, new_folder_path)
# print("NaiveBandedCuPy benchmarks done", flush=True)

# run_AMGX_benchmark(8, 8, 8, new_folder_path)
# run_AMGX_benchmark(16, 16, 16, new_folder_path)
# run_AMGX_benchmark(24, 24, 24, new_folder_path)
# run_AMGX_benchmark(32, 32, 32, new_folder_path)
# run_AMGX_benchmark(64, 64, 64, new_folder_path)
# run_AMGX_benchmark(128, 64, 64, new_folder_path)
# the following run out of memory
# run_AMGX_benchmark(128, 128, 64, new_folder_path)
# run_AMGX_benchmark(128, 128, 128, new_folder_path)

overall_end = time.time()

time_elapsed = overall_end - overall_start
minutes, seconds = divmod(time_elapsed, 60)

print(f"Benchmark took: {minutes} minutes and {seconds} seconds", flush=True)

print("****************************************************************************************", flush=True)
print("***********************************ALL BENCHMARKS DONE**********************************", flush=True)
print("****************************************************************************************", flush=True)