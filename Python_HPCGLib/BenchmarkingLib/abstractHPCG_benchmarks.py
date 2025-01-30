
import torch
import cupy as cp
import cupyx.scipy.sparse as cs
import numpy as np

from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import do_tests
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import num_bench_iterations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.abstractHPCG_tests as HPCG_tests
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseCuPy as BaseCuPy
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixConversions import csr_to_coo, banded_to_csr
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device

####################################################################################################
# coo torch benchmarks
def benchmark_CG_coo_torch(CGimplementation, timer:gpu_timer, A_coo: COOMatrix, A:torch.sparse.Tensor, r:torch.tensor, x:torch.tensor) -> None:

    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    if do_tests:
        baselineCG = BaseTorch.computeCG
        test_CG = HPCG_tests.test_CG_coo_torch(baselineCG, CGimplementation, A_coo, A, r, x)
        if not test_CG:
            print(f"ERROR: test failed for {CGimplementation.__name__} and size {nx}x{ny}x{nz}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        CGimplementation(nx, ny, nz, A, r, x, False)
        timer.stop_timer("computeCG")

def benchmark_MG_coo_torch(MGimplementation, timer:gpu_timer, A_coo: COOMatrix, A:torch.sparse.Tensor, r:torch.tensor, x:torch.tensor) -> None:

    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    if do_tests:
        baselineMG = BaseTorch.computeMG
        testresult_MG = HPCG_tests.test_MG(baselineMG, MGimplementation, A_coo = A_coo, A_torch = A, b_torch = r, x_torch = x)
        if not testresult_MG:
            print(f"ERROR: test failed for {MGimplementation.__name__} and size {nx}x{ny}x{nz}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        MGimplementation(nx, ny, nz, A, r, x, False)
        timer.stop_timer("computeMG")

def benchmark_SymGS_coo_torch(SymGSimplementation, timer:gpu_timer, A_coo: COOMatrix, A:torch.sparse.Tensor, r:torch.tensor, x:torch.tensor) -> None:

    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    if do_tests:
        baselineSymGS = BaseTorch.computeSymGS
        test_SymGS = HPCG_tests.test_symGS_coo_torch(SymGSimplementation, baselineSymGS, A_coo, A, r, x)
        if not test_SymGS:
            print(f"ERROR: test failed for {SymGSimplementation.__name__} and size {nx}x{ny}x{nz}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        SymGSimplementation(nx, ny, nz, A, r, x)
        timer.stop_timer("computeSymGS")

def benchmark_SPMV_torch(SPMVimplementation, timer:gpu_timer, A_coo: COOMatrix, A:torch.sparse.Tensor, x:torch.tensor, y:torch.tensor) -> None:

    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    if do_tests:

        baselineSPMV = BaseTorch.computeSPMV
        test_SPMV = HPCG_tests.test_SPMV_coo_torch(baselineSPMV, SPMVimplementation, A_coo, A, x, y)
        if not test_SPMV:
            print(f"ERROR: test failed for {SPMVimplementation.__name__} and size {nx}x{ny}x{nz}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        SPMVimplementation(nx, ny, nz, A, x, y)
        timer.stop_timer("computeSPMV")

def benchmark_WAXPBY_torch(WAXPBYimplementation, timer:gpu_timer, alpha: float, beta: float, x:torch.tensor, y:torch.tensor, w:torch.tensor) -> None:

    if do_tests:
        baselineWAXPBY = BaseTorch.computeWAXPBY
        test_WAXPBY = HPCG_tests.test_WAXPBY_torch(baselineWAXPBY, WAXPBYimplementation, alpha, beta, x, y, w)
        if not test_WAXPBY:
            print(f"ERROR: test failed for {WAXPBYimplementation.__name__}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        WAXPBYimplementation(alpha, x, beta, y, w)
        timer.stop_timer("computeWAXPBY")

def benchmark_dot_torch(dotimplementation, timer:gpu_timer, a:torch.tensor, b:torch.tensor, x:torch.tensor) -> None:

    if do_tests:
        baselineDot = BaseTorch.computeDot
        test_dot = HPCG_tests.test_dot_torch(baselineDot, dotimplementation, a, b,x)
        if not test_dot:
            print(f"ERROR: test failed for {dotimplementation.__name__}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        dotimplementation(a, b)
        timer.stop_timer("computeDot")
####################################################################################################
# csr cupy benchmarks

# def benchmark_CG_csr_cupy

def benchmark_SPMV_cupy(SPMVimplementation, timer:gpu_timer, A_csr: CSRMatrix, A_cupy:cs.csr_matrix, x_cupy:cp.ndarray, y_cupy:cp.ndarray) -> None:

    nx = A_csr.nx
    ny = A_csr.ny
    nz = A_csr.nz

    if do_tests:

        baselineSPMV = BaseTorch.computeSPMV

        A_coo = COOMatrix()
        csr_to_coo(A_csr, A_coo)
        A_torch = A_coo.to_torch()
        x_torch = torch.from_numpy(x_cupy.get()).to(device)
        y_torch = torch.from_numpy(y_cupy.get()).to(device)

        test_SPMV = HPCG_tests.test_SPMV(
            baselineSPMV=baselineSPMV, uutSPMV=SPMVimplementation,
            A_coo= A_coo, A_torch=A_torch, x_torch=x_torch, y_torch=y_torch,
            A_csr=A_csr, A_cupy_sparse=A_cupy, x_cupy=x_cupy, y_cupy=y_cupy)
        if not test_SPMV:
            print(f"ERROR: test failed for {SPMVimplementation.__name__} and size {nx}x{ny}x{nz}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        # we pass A_csr because the new implementations i.e. the cupy implementations pass metadata via that class.
        SPMVimplementation(A_csr, A_cupy, x_cupy, y_cupy)
        timer.stop_timer("computeSPMV")

def benchmark_SymGS_csr_cupy(SymGSimplementation, timer:gpu_timer, A_csr: CSRMatrix, A_cupy:cs.csr_matrix, x_cupy:cp.ndarray, y_cupy:cp.ndarray) -> None:

    nx = A_csr.nx
    ny = A_csr.ny
    nz = A_csr.nz

    original_x = x_cupy.copy()

    if do_tests:
        baselineSymGS = BaseTorch.computeSymGS

        A_coo = COOMatrix()
        csr_to_coo(A_csr, A_coo)
        A_torch = A_coo.to_torch()
        x_torch = torch.from_numpy(x_cupy.get()).to(device)
        y_torch = torch.from_numpy(y_cupy.get()).to(device)

        test_SymGS = HPCG_tests.test_symGS_csr_cupy(SymGSimplementation, baselineSymGS, A_coo, A_torch, x_torch, y_torch, A_csr, A_cupy, x_cupy, y_cupy)
        if not test_SymGS:
            print(f"ERROR: test failed for {SymGSimplementation.__name__} and size {nx}x{ny}x{nz}")
            return
    
    for i in range(num_bench_iterations):
        # because x gets changed we do need to reset it between runs
        x_cupy = original_x.copy()
        timer.start_timer()
        SymGSimplementation(A_csr, A_cupy, x_cupy, y_cupy)
        timer.stop_timer("computeSymGS")

    # greb the norm and store it
    norm = np.linalg.norm(cp.array(A_cupy @ x_cupy) - cp.array(y_cupy))

    timer.update_additional_info(timer.get_additional_info() + f"L2 Norm: {norm}")

    # reset the x vector
    x_cupy = original_x.copy()

####################################################################################################
# banded cupy benchmarks

def benchmark_SPMV_banded_cupy(SPMVimplementation, timer:gpu_timer, A_banded: BandedMatrix, A_dense: cp.ndarray, j_min_i_cupy: cp.ndarray, x_cupy:cp.ndarray, y_cupy: cp.ndarray) -> None:

    if do_tests:
        baselineSPMV = BaseCuPy.computeSPMV
        csr_Matrix = CSRMatrix()
        banded_to_csr(A_banded, csr_Matrix)
        A_sparse_cupy = csr_Matrix.get_cupy_matrix()

        test_SPMV = HPCG_tests.test_SPMV(
            baselineSPMV=baselineSPMV, uutSPMV=SPMVimplementation,
            A_csr=csr_Matrix, A_cupy_sparse=A_sparse_cupy,
            A_banded=A_banded, A_cupy_dense=A_dense, j_min_i_cupy=j_min_i_cupy,
            x_cupy=x_cupy, y_cupy=y_cupy
        )

        if not test_SPMV:
            print(f"ERROR: test failed for {SPMVimplementation.__name__} and size {A_banded.nx}x{A_banded.ny}x{A_banded.nz}")
            return
    
    for i in range(num_bench_iterations):
        timer.start_timer()
        SPMVimplementation(A_banded, j_min_i_cupy, A_dense, x_cupy, y_cupy)
        timer.stop_timer("computeSPMV")
