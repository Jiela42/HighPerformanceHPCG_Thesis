
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixConversions import coo_to_csr
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import error_tolerance
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import cg_error_tolerance
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode

from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import read_dimension
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import print_differeing_vectors

import os
import torch
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
from typing import Optional
##############################################################################################################################
# COO Torch Versions
##############################################################################################################################
def test_CG_coo_torch(baselineCG, uutCG, A_coo: COOMatrix, A:torch.sparse.Tensor, r:torch.tensor, x:torch.tensor) -> bool:
    
    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    baselineCG(nx, ny, nz, A, r, x, False)
    base_x = x.clone()
    uutCG(nx, ny, nz, A, r, x, False)
    uut_x = x.clone()

    if not torch.allclose(base_x, uut_x, atol=cg_error_tolerance):
        print_differeing_vectors(uut_x, base_x, 5)

    return torch.allclose(base_x, uut_x, atol=cg_error_tolerance)

def test_symGS_coo_torch(uutSymGS, baselineSymGS, A_coo: COOMatrix, A:torch.sparse.Tensor, r:torch.tensor, x:torch.tensor) -> bool:
    
    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    x.zero_()
    baselineSymGS(nx, ny, nz, A, r, x)
    base_x = x.clone()
    x.zero_()
    uutSymGS(nx, ny, nz, A, r, x)
    uut_x = x.clone()

    return torch.allclose(base_x, uut_x, atol=error_tolerance)

def test_SPMV_coo_torch(uutSPMV, baselineSPMV, A_coo: COOMatrix, A:torch.sparse.Tensor, x:torch.tensor, y:torch.tensor) -> bool:
    
    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    baselineSPMV(nx, ny, nz, A, x, y)
    base_y = y.clone()
    uutSPMV(nx, ny, nz, A, x, y)
    uut_y = y.clone()

    return torch.allclose(base_y, uut_y, atol=error_tolerance)

def test_WAXPBY_torch(uutWAXPBY, baselineWAXPBY, alpha: float, beta: float, x:torch.tensor, y:torch.tensor, w:torch.tensor) -> bool:
        
    w.zero_()
    baselineWAXPBY(alpha, x, beta, y, w)
    base_w = w.clone()
    w.zero_()
    uutWAXPBY(alpha, x, beta, y, w)
    uut_w = w.clone()

    return torch.allclose(base_w, uut_w, atol=error_tolerance)

def test_dot_torch(uutDot, baselineDot, a:torch.tensor, b:torch.tensor, x:torch.tensor) -> bool:
    
    base_x = baselineDot(a, b)
    uut_x = uutDot(a, b)

    return torch.allclose(base_x, uut_x, atol=error_tolerance)

def test_MG_from_file(uutMG) -> bool:
    # this one is special because we test against what we have on file.

    all_tests_passed = True

    path_to_testcases = "../../hpcg_output"

    sub_directories = [d for d in os.listdir(path_to_testcases)]

    for dir in sub_directories:
        dir_path = os.path.join(path_to_testcases, dir)
        dimensions_path = os.path.join(dir_path, "dimA.txt")
        b_computed_path = os.path.join(dir_path, "b_computed.txt")
        x_overlap_path = os.path.join(dir_path, "x_overlap.txt")
        x_overlap_after_mg_path = os.path.join(dir_path, "x_overlap_after_mg.txt")
        dimensions_dict = read_dimension(dimensions_path)
        nx = dimensions_dict["nx"]
        ny = dimensions_dict["ny"]
        nz = dimensions_dict["nz"]

        _, A, y = generations.generate_torch_coo_problem(nx, ny, nz)

        b_computed = torch.tensor(np.loadtxt(b_computed_path), device=device, dtype=torch.float64)
        x_overlap = torch.tensor(np.loadtxt(x_overlap_path), device=device, dtype=torch.float64)
        x_overlap_after_mg = torch.tensor(np.loadtxt(x_overlap_after_mg_path), device=device, dtype=torch.float64)

        # we only test baseTorch with this minitest, every other version of MG is compared to BaseTorch
        empty_x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)
        uutMG(nx, ny, nz, A, b_computed, empty_x, 0)
        x_BaseTorch = empty_x.clone()

        all_tests_passed = all_tests_passed and torch.allclose(x_BaseTorch, x_overlap_after_mg, atol=error_tolerance)

    return all_tests_passed

def test_MG_against_BaseLine_torch_coo(
            uutMG, baselineMG,
            A_coo: COOMatrix, A_torch: torch.sparse.Tensor, b_torch: torch.tensor, x_torch: torch.tensor
        ) -> bool:
    
    nx = A_coo.nx
    ny = A_coo.ny
    nz = A_coo.nz

    empty_x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)
    uutMG(nx, ny, nz, A_torch, b_torch, empty_x, 0)
    x_uut = empty_x.clone()

    empty_x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)
    baselineMG(nx, ny, nz, A_torch, b_torch, empty_x, 0)
    x_baseline = empty_x.clone()

    return torch.allclose(x_uut, x_baseline, atol=error_tolerance)
##############################################################################################################################
# CSR Cupy Versions
##############################################################################################################################
def test_SPMV_csr_cupy(uut, baseline, A_csr: CSRMatrix, A: sp.csr_matrix, x: cp.ndarray, y: cp.ndarray) -> bool:
    
    nx = A_csr.nx
    ny = A_csr.ny
    nz = A_csr.nz

    baseline(nx, ny, nz, A, x, y)
    base_y = y.copy()
    uut(nx, ny, nz, A, x, y)
    uut_y = y.copy()

    return cp.allclose(base_y, uut_y, atol=error_tolerance)

##############################################################################################################################
# baseline torch coo, unit under test cupy csr
##############################################################################################################################
def test_SPMV_csr_cupy_coo_torch(
        uutSPMV, baselineSPMV,
        A_csr: CSRMatrix, A_cupy: sp.csr_matrix, x_cupy: cp.ndarray, y_cupy: cp.ndarray,
        A_coo: COOMatrix, A_torch:torch.sparse.Tensor, x_torch: torch.tensor, y_torch: torch.tensor
) -> bool:
    
    nx = A_csr.nx
    ny = A_csr.ny
    nz = A_csr.nz

    baselineSPMV(nx, ny, nz, A_torch, x_torch, y_torch)
    base_y = y_torch.clone()
    # the newer implementations just take a matrix and no dimensions (or any meta data)
    uutSPMV(A_csr, A_cupy, x_cupy, y_cupy)
    # print(f"y_cupy first 5: {y_cupy[:5]}")
    uut_y = torch.tensor(y_cupy.get(), device=device, dtype=torch.float64)

    test_result = torch.allclose(base_y, uut_y, atol=error_tolerance)
    
    if not test_result:
        print_differeing_vectors(uut_y, base_y, 5)

    return test_result

##############################################################################################################################

# This function is used to select the right version to test the CG method
def test_CG(baselineCG, uutCG,
            A_coo: Optional[COOMatrix] = None, A_csr: Optional[CSRMatrix] = None, A_banded: Optional[BandedMatrix] = None,
            A_torch: Optional[torch.sparse.Tensor] = None, r_torch: Optional[torch.tensor] = None, x_torch: Optional[torch.tensor] = None,
            ) -> bool:

    if A_coo is not None:
        if A_torch is not None and r_torch is not None and x_torch is not None:
            return test_CG_coo_torch(baselineCG, uutCG, A_coo, A_torch, r_torch, x_torch)
        else:
            if developer_mode:
                print("ERROR: There is no version to test CG where the matrix is COO and the vectors are not torch tensors")
            return False
    # if A_csr is not None:
    #     test_CG_csr(baselineCG, uutCG, A_csr, A, r, x, depth)
    # if A_banded is not None:
    #     test_CG_banded(baselineCG, uutCG, A_banded, A, r, x, depth)

def test_MG(
        uutMG, baselineMG = None,
        A_coo: Optional[COOMatrix] = None, A_csr: Optional[CSRMatrix] = None, A_banded: Optional[BandedMatrix] = None,
        A_torch: Optional[torch.sparse.Tensor] = None, b_torch: Optional[torch.tensor] = None, x_torch: Optional[torch.tensor] = None,
            ) -> bool:
    # we test the MG method against the baseline (if provided) and against the file outputs we have 
    all_tests_passed = True
    test_from_file = test_MG_from_file(uutMG)

    if not test_from_file:
        all_tests_passed = False
        print("The MG test from file failed", flush=True)
    
    test_against_bl = True

    if baselineMG is not None:
        if A_coo is not None:
            test_against_bl = test_MG_against_BaseLine_torch_coo(uutMG, baselineMG, A_coo, A_torch, b_torch, x_torch)
        else:
            if developer_mode:
                print("ERROR: There is no version to test MG against baseline where the matrix is not COO and the vectors are not torch tensors")
            return False
        if not test_against_bl:
            all_tests_passed = False
            print("The MG test against baseline failed", flush=True)

    return all_tests_passed

def test_symGS(
        uutSymGS, baselineSymGS,
        A_coo: Optional[COOMatrix] = None, A_csr: Optional[CSRMatrix] = None, A_banded: Optional[BandedMatrix] = None,
        A_torch: Optional[torch.sparse.Tensor] = None, r_torch: Optional[torch.tensor] = None, x_torch: Optional[torch.tensor] = None,
        ) -> bool:
    
    if A_coo is not None:
        if A_torch is not None and r_torch is not None and x_torch is not None:
            return test_symGS_coo_torch(uutSymGS, baselineSymGS, A_coo, A_torch, r_torch, x_torch)
        else:
            if developer_mode:
                print("ERROR: There is no version to test SymGS where the matrix is COO and the vectors are not torch tensors")
            return False

def test_SPMV(
        uutSPMV, baselineSPMV,
        A_coo: Optional[COOMatrix] = None, A_csr: Optional[CSRMatrix] = None, A_banded: Optional[BandedMatrix] = None,
        A_torch: Optional[torch.sparse.Tensor] = None, x_torch: Optional[torch.tensor] = None, y_torch: Optional[torch.tensor] = None,
        A_cupy: Optional[sp.csr_matrix] = None, x_cupy: Optional[cp.ndarray] = None, y_cupy: Optional[cp.ndarray] = None
        ) -> bool:
    

    torch_implementation = A_torch is not None and x_torch is not None and y_torch is not None
    cupy_implementation = A_cupy is not None and x_cupy is not None and y_cupy is not None

    if A_csr is not None and A_coo is not None:
        # this means we have a mixed implementation (the baseline is COO, the unit under test is CSR)
        if cupy_implementation and torch_implementation:
            return test_SPMV_csr_cupy_coo_torch(
                uutSPMV, baselineSPMV,
                A_csr, A_cupy, x_cupy, y_cupy,
                A_coo, A_torch, x_torch, y_torch
                )
        else:
            if developer_mode:
                print("ERROR: There is no version to test SPMV where the matrix is CSR and the vectors are not cupy arrays")
            return False

    elif A_csr is not None:
        if cupy_implementation:
            return test_SPMV_csr_cupy(uutSPMV, baselineSPMV, A_csr, A_cupy, x_cupy, y_cupy)
        else:
            if developer_mode:
                print("ERROR: There is no version to test SPMV where the matrix is CSR and the vectors are not cupy arrays")
            return False

    elif A_coo is not None:
        if torch_implementation:
            return test_SPMV_coo_torch(uutSPMV, baselineSPMV, A_coo, A_torch, x_torch, y_torch)
        else:
            if developer_mode:
                print("ERROR: There is no version to test SPMV where the matrix is COO and the vectors are not torch tensors")
            return False
        
def test_WAXPBY(
        uutWAXPBY, baselineWAXPBY,
        alpha: float, beta: float,
        x_torch: Optional[torch.tensor] = None, y_torch: Optional[torch.tensor] = None, w_torch: Optional[torch.tensor] = None
        ) -> bool:
    
    torch_implementation = x_torch is not None and y_torch is not None and w_torch is not None

    if torch_implementation:
        return test_WAXPBY_torch(uutWAXPBY, baselineWAXPBY, alpha, beta, x_torch, y_torch, w_torch)

def test_dot(
    uutDot, baselineDot,
    a_torch: Optional[torch.tensor] = None, b_torch: Optional[torch.tensor] = None, x_torch: Optional[torch.Tensor] = None
    ) -> bool:

    torch_implementation = a_torch is not None and b_torch is not None and x_torch is not None

    if torch_implementation:
        return test_dot_torch(uutDot, baselineDot, a_torch, b_torch, x_torch)
    



