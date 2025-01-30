import torch
import cupyx as cp

from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import ault_node
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import max_dim_size, limit_matrix_size
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import limit_matrix_size_for_cg, max_dim_cg
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseCuPy as BaseCuPy
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.abstractHPCG_benchmarks import *



def run_BaseCuPy_3d27pt_benchmark(nx: int, ny: int, nz: int, save_folder) -> None:
    A_csr, A_cupy, y_cupy = generations.generate_cupy_csr_problem(nx, ny, nz)
    x_cupy = cp.zeros(nx*ny*nz, dtype=cp.float64)

    a,b = cp.random.rand(nx*ny*nz, dtype=cp.float64), cp.random.rand(nx*ny*nz, dtype=cp.float64)

    alpha, beta = cp.random.rand(), cp.random.rand()

    nnz = A_csr.nnz

    version_name = "BaseCuPy"

    matrix_timer = gpu_timer(
        version_name = version_name,
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nnz,
        folder_path = save_folder
    )

    vector_timer = gpu_timer(
        version_name = version_name,
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nx*ny*nz,
        folder_path = save_folder
    )

    # if limit_matrix_size and nx < max_dim_size and ny < max_dim_size and nz < max_dim_size:
    #     benchmark_CG_csr_cupy(CGimplementation=BaseCuPy.computeCG, timer=matrix_timer, A_csr=A_csr, A=A_cupy, r=y_cupy, x=x_cupy)
    #     benchmark_MG_csr_cupy(BaseCuPy.computeMG, matrix_timer, A_csr, A_cupy, y_cupy, x_cupy)
    

    # benchmark_SPMV_cupy(BaseCuPy.computeSPMV, matrix_timer, A_csr, A_cupy, x_cupy, y_cupy)
    # benchmark_WAXPBY_cupy(BaseCuPy.computeWAXPBY, vector_timer, alpha=alpha, x=a, beta=beta, y=b, w=x_cupy)
    # benchmark_dot_cupy(BaseCuPy.computbeDot, vector_timer, a, b, x_cupy)

    matrix_timer.destroy_timer()
    vector_timer.destroy_timer()



    # the BaseCuPy contains three different versions of the SymGS algorithm so we need to benchmark all of them
    symGS_Implementations_Names = [
        (BaseCuPy.computeSymGS_minres, "CuPy 5 iterations (minres)"),
        (BaseCuPy.computeSymGS_lsmr, "CuPy 5 iterations (lsmr)"),
        (BaseCuPy.computeSymGS_gmres, "CuPy 5 iterations (gmres)")
    ]

    for symGS_Implementation, symGS_name in symGS_Implementations_Names:

        # get a new matrix timer each time
        matrix_timer = gpu_timer(
            version_name = symGS_name,
            ault_node = ault_node,
            matrix_type = "3d_27pt",
            nx = nx,
            ny = ny,
            nz = nz,
            nnz = nnz,
            folder_path = save_folder
        )

        benchmark_SymGS_csr_cupy(symGS_Implementation, matrix_timer, A_csr, A_cupy, x_cupy, y_cupy)

        matrix_timer.destroy_timer()



def run_BaseCuPy_benchmark(nx:int, ny:int, nz:int, save_folder):
    
    run_BaseCuPy_3d27pt_benchmark(nx, ny, nz, save_folder)