import sys
import os



import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg
import cupyx
import torch
import numpy as np
from typing import Tuple

# my personal implementation of generations for the matrices, vectors etc.
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
import HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.generations as generations
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch

from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device

version_name = "BaseCuPy"

num_pre_smoother_steps = 1
num_post_smoother_steps = 1
tolerance = 0.0
max_iter = 50
debug = True # this will also skip the preconditioning in the CG function

def computeDot(x: torch.tensor, y: torch.tensor) -> float:
    print(f"WARNING: computeDot not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeDot(x, y)
    

def computeSymGS_minres(A_csr: CSRMatrix,
                A: sp.csr_matrix, x: cp.ndarray, y: cp.ndarray)-> int:
    
    solution = cupyx.scipy.sparse.linalg.minres(A, b=y, x0=x, maxiter=5)

    # if solution[1] != 0:
    #     print(f"WARNING: GMRES did not converge, error code: {solution[1]}")

    cp.copyto(x, solution[0])
    return 0

def computeSymGS_lsmr(A_csr: CSRMatrix,
                A: sp.csr_matrix, x: cp.ndarray, y: cp.ndarray)-> int:
    
    solution = cupyx.scipy.sparse.linalg.lsmr(A, b=y, x0=x, maxiter=5)
    cp.copyto(x, solution[0])
    return 0

def computeSymGS_gmres(A_csr: CSRMatrix,
                A: sp.csr_matrix, x: cp.ndarray, y: cp.ndarray)-> int:
    solution = cupyx.scipy.sparse.linalg.gmres(A, b=y, x0=x, maxiter=5)
    cp.copyto(x, solution[0])
    return 0

def computeSPMV(A_csr: CSRMatrix,
                A: sp.csr_matrix, x: cp.ndarray, y: cp.ndarray)-> int:
    
    # print(f"type y: {type(y)}")
    # print(f"type x: {type(x)}")
    # print(f"type A: {type(A)}")
    # print(f"type Adotx: {type(A.dot(x))}")
    # cp.dot(A, x, y)
    # A.dot(x)
    cp.copyto(y, A.dot(x))

    return 0
    
def computeRestriction(Afx: torch.Tensor, rf: torch.Tensor,
                       nc: int, f2c: torch.Tensor, rc: torch.Tensor)-> int:
    
    rc[:] = rf[f2c[:nc]] - Afx[f2c[:nc]]
    
    return 0

def computeProlongation(xf: torch.Tensor, xc: torch.Tensor, f2c: torch.Tensor, nc: int)-> int:
    
    xf[f2c[:]] += xc[:nc]
    
    return 0

def computeMG(nx: int, nz: int, ny: int,
              A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor,
              depth: int)-> int:
    
    """
    Computes the Multigrid (MG) method.

    Parameters:
    nx [in] (int): Number of grid points in the x-direction.
    ny [in] (int): Number of grid points in the y-direction.
    nz [in] (int): Number of grid points in the z-direction.
    A [in] (torch.sparse.Tensor): The sparse matrix representing the system.
    r [in] (torch.Tensor): The residual vector.
    x [inout] (torch.Tensor): The solution vector.
    depth (int): The current depth of the multigrid hierarchy.

    Returns:
    int: Status code (0 for success).
    """
    print(f"WARNING: computeMG not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeMG(nx, nz, ny, A, r, x, depth)

def computeWAXPBY(a: float, x: torch.Tensor, b: float, y: torch.Tensor, w: torch.Tensor)-> int:
    # note that double can also be x or y!
    print(f"WARNING: computeWAXPBY not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeWAXPBY(a, x, b, y, w)

def computeCG(nx: int, ny: int, nz: int,
              A: torch.sparse.Tensor, y: torch.Tensor, x: torch.Tensor) -> int:
    print(f"WARNING: computeCG not implemented for {version_name}, using BaseTorch implementation")

