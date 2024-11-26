import torch
import random
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import ault_node
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import limit_matrix_size_for_cg, max_dim_cg
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import limit_matrix_size, max_dim_size
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.matlab_reference as MatlabReference
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.abstractHPCG_benchmarks import *


def run_MatlabReference_3d27pt_benchmark(nx: int, ny: int, nz: int, save_folder) -> None:
    A_coo,A,y = generations.generate_torch_coo_problem(nx, ny, nz)
    x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)
    
    a,b = torch.rand(nx*ny*nz, device=device, dtype=torch.float64), torch.rand(nx*ny*nz, device=device, dtype=torch.float64)
    alpha, beta = random.random(), random.random()

    nnz = A._nnz()

    matrix_timer = gpu_timer(
        version_name = "MatlabReference",
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nnz,
        folder_path = save_folder
    )

    vector_timer = gpu_timer(
        version_name = "MatlabReference",
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nx*ny*nz,
        folder_path = save_folder
    )

    # we also include checks to not run the super big matrices
    # if limit_matrix_size_for_cg and nx < max_dim_cg and ny < max_dim_cg and nz < max_dim_cg:

    if limit_matrix_size and nx < max_dim_size and ny < max_dim_size and nz < max_dim_size:
        benchmark_SymGS_coo_torch(MatlabReference.computeSymGS, matrix_timer, A_coo, A, y, x)
        benchmark_CG_coo_torch(CGimplementation=MatlabReference.computeCG, timer=matrix_timer, A_coo=A_coo, A=A, r=y, x=x)
    
    benchmark_SPMV_torch(MatlabReference.computeSPMV, matrix_timer, A_coo, A, x, y)
    benchmark_WAXPBY_torch(MatlabReference.computeWAXPBY, vector_timer, alpha=alpha, x=a, beta=beta, y=b, w=x)
    benchmark_dot_torch(MatlabReference.computeDot, vector_timer, a, b, x)

    matrix_timer.destroy_timer()
    vector_timer.destroy_timer()

def run_MatlabReference_benchmark(nx: int, ny: int, nz: int, save_folder) -> None:

    run_MatlabReference_3d27pt_benchmark(nx, ny, nz, save_folder)
    