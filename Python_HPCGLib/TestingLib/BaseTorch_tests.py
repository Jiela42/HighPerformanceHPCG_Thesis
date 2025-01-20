import torch

import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.matlab_reference as matlab_reference
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import limit_matrix_size_for_cg, max_dim_cg

import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.abstractHPCG_tests as abstract_tests

def test_BaseTorch_3d27p_on_random_matrix(A_coo: COOMatrix, A_torch: torch.sparse.Tensor, y: torch.Tensor) -> bool:
    # we use this function since we have requirements for the matrix for some of the test cases.
    # essentially we only test spmv here

    all_tests_passed = True
    
    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    a = torch.rand(nx*ny*nz, device=device, dtype=torch.float64)
    x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)

    baselineSPMV = matlab_reference.computeSPMV
    testSPMV = BaseTorch.computeSPMV

    SPMV_test = abstract_tests.test_SPMV(
        baselineSPMV=baselineSPMV, uutSPMV=testSPMV,
        A_coo=A_coo, A_torch=A_torch, x_torch = a, y_torch = x)
    
    if not SPMV_test:
        the_test_passed = False
        print(f"The BaseTorch SPMV test failed for size: {nx}x{ny}x{nz}", flush=True)
    

    return all_tests_passed

def test_BaseTorch_3d27p_on_matrix(A_coo: COOMatrix, A_torch: torch.sparse.Tensor, y: torch.Tensor) -> bool:
    
    all_tests_passed = True

    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz
    
    x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)
    a,b = torch.rand(nx*ny*nz, device=device, dtype=torch.float64), torch.rand(nx*ny*nz, device=device, dtype=torch.float64)
    alpha, beta = torch.rand(1).item(), torch.rand(1).item()

    testCG = BaseTorch.computeCG
    testMG = BaseTorch.computeMG
    testSymGS = BaseTorch.computeSymGS
    testSPMV = BaseTorch.computeSPMV
    testWAXPBY = BaseTorch.computeWAXPBY
    testDot = BaseTorch.computeDot

    baselineCG = matlab_reference.computeCG
    baselineSymGS = matlab_reference.computeSymGS
    baselineSPMV = matlab_reference.computeSPMV
    baselineWAXPBY = matlab_reference.computeWAXPBY
    baselineDot = matlab_reference.computeDot

    # now we call the abstract test methods
    if limit_matrix_size_for_cg and nx > max_dim_cg and ny > max_dim_cg and nz > max_dim_cg:
        CG_test = abstract_tests.test_CG(
            baselineCG=baselineCG, uutCG=testCG,
            A_coo=A_coo, A_torch=A_torch, r_torch = y, x_torch = x)

        if not CG_test:
            all_tests_passed = False
            print(f"The BaseTorch CG test failed for size: {nx}x{ny}x{nz}", flush=True)
        elif developer_mode:
            print(f"The BaseTorch CG test passed for size: {nx}x{ny}x{nz}", flush=True)


    # this is just here for a reference for further files
    # MG_base_test = abstract_tests.test_MG(
    #     uutMG=testMG, baselineMG=testMG,
    #     A_coo=A_coo, A_torch=A_torch, b_torch=y, x_torch=x)
    
    # if not MG_base_test:
    #     all_tests_passed = False
    #     print(f"The BaseTorch MG test failed", flush=True)

    MG_test = abstract_tests.test_MG(uutMG=testMG)

    if not MG_test:
        all_tests_passed = False
        print(f"The BaseTorch MG test failed for size {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseTorch MG test passed for size {nx}x{ny}x{nz}", flush=True)
    
    if nx < 32 and ny < 32 and nz < 32:
        SymGS_test = abstract_tests.test_symGS(
            baselineSymGS=baselineSymGS, uutSymGS=testSymGS,
            A_coo=A_coo, A_torch=A_torch, r_torch = y, x_torch = x)
        
        if not SymGS_test:
            all_tests_passed = False
            print(f"The BaseTorch SymGS test failed for size: {nx}x{ny}x{nz}", flush=True)
        elif developer_mode:
            print(f"The BaseTorch SymGS test passed for size: {nx}x{ny}x{nz}", flush=True)

    SPMV_test = abstract_tests.test_SPMV(
        baselineSPMV=baselineSPMV, uutSPMV=testSPMV,
        A_coo=A_coo, A_torch=A_torch, x_torch = a, y_torch = x)
    
    if not SPMV_test:
        all_tests_passed = False
        print(f"The BaseTorch SPMV test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseTorch SPMV test passed for size: {nx}x{ny}x{nz}", flush=True)

    WAXPBY_test = abstract_tests.test_WAXPBY(
        baselineWAXPBY=baselineWAXPBY, uutWAXPBY=testWAXPBY,
        alpha=alpha, x_torch=a, beta=beta, y_torch=b, w_torch=x)
    
    if not WAXPBY_test:
        all_tests_passed = False
        print(f"The BaseTorch WAXPBY test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseTorch WAXPBY test passed for size: {nx}x{ny}x{nz}", flush=True)

    dot_test = abstract_tests.test_dot(
        baselineDot=baselineDot, uutDot=testDot,
        a_torch=a, b_torch=b, x_torch=x)
    
    if not dot_test:
        all_tests_passed = False
        print(f"The BaseTorch Dot test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The BaseTorch Dot test passed for size: {nx}x{ny}x{nz}", flush=True)

    return all_tests_passed

def test_BaseTorch_3d27p(nx:int, ny:int, nz:int) -> bool:
    all_tests_passed = True

    A_coo, A, y = generations.generate_torch_coo_problem(nx, ny, nz)
    # print(y)

    current_matrix_test = test_BaseTorch_3d27p_on_matrix(A_coo, A, y)
    
    if not current_matrix_test:
        all_tests_passed = False
        print(f"BaseTorch on standard HPCG Matrix failed for size: {nx}x{ny}x{nz}", flush=True)

    # we now also test for iterative values
    A_coo.iterative_values()
    A = A_coo.to_torch()

    current_matrix_test = test_BaseTorch_3d27p_on_random_matrix(A_coo, A, y)
    if not current_matrix_test:
        all_tests_passed = False
        print(f"BaseTorch on random HPCG Matrix failed for size: {nx}x{ny}x{nz}", flush=True)

    
    return all_tests_passed


# again, we may need to change this when we get more matrix types
def test_BaseTorch(nx:int, ny:int, nz:int) -> bool:

    all_tests_passed = True
    all_tests_passed = all_tests_passed and test_BaseTorch_3d27p(nx, ny, nz)

    return all_tests_passed

def get_L2_norm_SymGS(nx:int, ny:int, nz:int):


    A_coo, A, y = generations.generate_torch_coo_problem(nx, ny, nz)
    x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)

    BaseTorch.computeSymGS(nx, ny, nz, A, x, y)

    import numpy as np
    import cupy as cp

    Ax = A @ x
    norm = np.linalg.norm(cp.array(Ax) - cp.array(y))

    print(f"{nx}x{ny}x{nz}: {norm}", flush=True)


