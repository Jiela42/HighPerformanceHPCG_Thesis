import cupy as cp

from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import ault_node

from HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions import NaiveBandedCuPy
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.abstractHPCG_benchmarks import *

def run_NaiveBandedCuPy_3d27pt_benchmark(nx:int, ny:int, nz:int, save_folder):

    A_banded, A_cupy_dense, y_cupy = generations.generate_cupy_banded_problem(nx, ny, nz)
    a,b = cp.random.rand(nx*ny*nz, dtype=cp.float64), cp.random.rand(nx*ny*nz, dtype=cp.float64)
    j_min_i_cupy = cp.array(A_banded.j_min_i, dtype=cp.int32)

    alpha, beta = cp.random.rand(), cp.random.rand()

    nnz = A_banded.nnz

    matrix_timer = gpu_timer(
        version_name = "NaiveBanded CuPy",
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nnz,
        folder_path = save_folder
    )

    vector_timer = gpu_timer(
        version_name = "NaiveBanded CuPy",
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nx*ny*nz,
        folder_path = save_folder
    )

    benchmark_SPMV_banded_cupy(NaiveBandedCuPy.computeSPMV, matrix_timer, A_banded, A_cupy_dense, j_min_i_cupy, a, y_cupy)

    matrix_timer.destroy_timer()
    vector_timer.destroy_timer()

def run_NaiveBandedCuPy_benchmark(nx:int, ny:int, nz:int, save_folder):

    run_NaiveBandedCuPy_3d27pt_benchmark(nx, ny, nz, save_folder)