import torch
import cupy as cp
import cupyx.scipy.sparse as sp

import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseCuPy as BaseCuPy
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixConversions import csr_to_coo
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode

import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.abstractHPCG_tests as abstract_tests

def test_BaseCuPy_3d27p_on_random_matrix(A_csr: CSRMatrix, A_cupy: sp.csr_matrix) -> bool:
    # we use this function since we have requirements for the matrix for some of the test cases.
    # essentially we only test spmv here

    all_tests_passed = True

    A_coo = COOMatrix()
    csr_to_coo(A_csr, A_coo)

    nx = A_csr.nx
    ny = A_csr.ny
    nz = A_csr.nz

    x = cp.zeros(A_csr.num_cols, dtype=cp.float64)
    a = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    # a = cp.ones(A_csr.num_cols, dtype=cp.float64)
    b = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    alpha, beta = cp.random.rand(1).item(), cp.random.rand(1).item()
    
    x_torch = torch.zeros(A_csr.num_cols, device=device, dtype=torch.float64)
    a_torch = torch.tensor(a.get(), device=device, dtype=torch.float64)
    b_torch = torch.tensor(b.get(), device=device, dtype=torch.float64)
    A_torch = A_coo.get_torch_matrix()

    testSPMV = BaseCuPy.computeSPMV    
    baselineSPMV = BaseTorch.computeSPMV

    # now we call the abstract test methods
    
    SPMV_test = abstract_tests.test_SPMV(
        baselineSPMV=baselineSPMV, uutSPMV=testSPMV,
        A_coo=A_coo, A_torch=A_torch, x_torch = a_torch, y_torch = x_torch,
        A_csr=A_csr, A_cupy_sparse=A_cupy, x_cupy = a, y_cupy = x)
    
    if not SPMV_test:
        all_tests_passed = False
        print(f"The BaseCuPy SPMV test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseCuPy SPMV test passed for size: {nx}x{ny}x{nz}", flush=True)

    return all_tests_passed

def test_BaseCuPy_3d27p_on_matrix(A_csr: CSRMatrix, A_cupy: sp.csr_matrix) -> bool:
    
    all_tests_passed = True

    A_coo = COOMatrix()
    csr_to_coo(A_csr, A_coo)

    nx = A_csr.nx
    ny = A_csr.ny
    nz = A_csr.nz

    y = generations.generate_y_forHPCG_problem(nx, ny, nz)
    y_cupy = cp.array(y, dtype=cp.float64)
    y_torch = torch.tensor(y, device=device, dtype=torch.float64)

    x = cp.zeros(A_csr.num_cols, dtype=cp.float64)
    a = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    b = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    alpha, beta = cp.random.rand(1).item(), cp.random.rand(1).item()
    
    x_torch = torch.zeros(A_csr.num_cols, device=device, dtype=torch.float64)
    a_torch = torch.tensor(a.get(), device=device, dtype=torch.float64)
    b_torch = torch.tensor(b.get(), device=device, dtype=torch.float64)
    A_torch = A_coo.get_torch_matrix()


    testSPMV = BaseCuPy.computeSPMV    
    baselineSPMV = BaseTorch.computeSPMV

    # now we call the abstract test methods
    
    SPMV_test = abstract_tests.test_SPMV(
        baselineSPMV=baselineSPMV, uutSPMV=testSPMV,
        A_coo=A_coo, A_torch=A_torch, x_torch = a_torch, y_torch = x_torch,
        A_csr=A_csr, A_cupy_sparse=A_cupy, x_cupy = a, y_cupy = x)
    
    if not SPMV_test:
        all_tests_passed = False
        print(f"The BaseCuPy SPMV test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseCuPy SPMV test passed for size: {nx}x{ny}x{nz}", flush=True)

    SymGS_test = abstract_tests.test_SymGS(
        uutSymGS=BaseCuPy.computeSymGS, baselineSymGS=BaseTorch.computeSymGS,
        A_csr=A_csr, A_cupy_sparse=A_cupy, x_cupy = x, y_cupy = y_cupy,
        A_coo=A_coo, A_torch=A_torch, x_torch = x_torch, y_torch = y_torch 
        )
    
    if not SymGS_test:
        all_tests_passed = False
        print(f"The BaseCuPy SymGS test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseCuPy SymGS test passed for size: {nx}x{ny}x{nz}", flush=True)


    return all_tests_passed

def test_BaseCuPy_3d27p(nx:int, ny:int, nz:int) -> bool:
    all_tests_passed = True

    A_csr, A, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    # print(y)

    current_matrix_test = test_BaseCuPy_3d27p_on_matrix(A_csr, A)
    
    if not current_matrix_test:
        all_tests_passed = False
        print(f"BaseCuPy on standard HPCG Matrix failed for size: {nx}x{ny}x{nz}", flush=True)

    # # we now also test for iterative values
    # A_csr.iterative_values()
    # A = A_csr.to_cupy()

    # current_matrix_test = test_BaseCuPy_3d27p_on_random_matrix(A_csr, A)
    # if not current_matrix_test:
    #     all_tests_passed = False
    #     print(f"BaseCuPy on random HPCG Matrix failed for size: {nx}x{ny}x{nz}", flush=True)

    
    return all_tests_passed


# again, we may need to change this when we get more matrix types
def test_BaseCuPy(nx:int, ny:int, nz:int) -> bool:

    all_tests_passed = True
    all_tests_passed = all_tests_passed and test_BaseCuPy_3d27p(nx, ny, nz)

    return all_tests_passed

