
import torch

from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import do_tests
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import num_bench_iterations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.abstractHPCG_tests as HPCG_tests
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix

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