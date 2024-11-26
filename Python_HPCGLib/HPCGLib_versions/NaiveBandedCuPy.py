import sys
import os



import cupy as cp
from cupyx import jit
import cupyx.scipy.sparse as sp
import torch
import numpy as np
import math
from typing import Tuple

# my personal implementation of generations for the matrices, vectors etc.
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
import HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.generations as generations
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch

from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import device

version_name = "Naive Banded CuPy"

num_pre_smoother_steps = 1
num_post_smoother_steps = 1
tolerance = 0.0
max_iter = 50
debug = True # this will also skip the preconditioning in the CG function

def computeDot(x: torch.tensor, y: torch.tensor) -> float:
    print(f"WARNING: computeDot not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeDot(x, y)
    

def computeSymGS(nx: int, nz: int, ny: int,
                 A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor)-> int:
    print(f"WARNING: computeSymGS not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeSymGS(nx, nz, ny, A, r, x)

@jit.rawkernel()
def SPMV_kernel(
    num_rows: int, num_cols: int, num_bands: int,
    j_min_i: cp.ndarray,
    A_banded: cp.ndarray, x: cp.ndarray, y: cp.ndarray
    ):
    
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    # every thread computes one or more rows of the matrix
    for i in range(tid, num_rows, jit.blockDim.x * jit.gridDim.x):
        sum_i = 0.0
        for band in range(num_bands):
            j = i + j_min_i[band]
            if j >= 0 and j < num_cols:
                sum_i += A_banded[i*num_bands + band] * x[j]

        y[i] = sum_i

def computeSPMV(A_banded: BandedMatrix,
                A: sp.csr_matrix, x: cp.ndarray, y: cp.ndarray)-> int:
    
    num_rows = A_banded.num_rows
    num_cols = A_banded.num_cols
    num_bands = A_banded.num_bands

    j_min_i = cp.array(A_banded.j_min_i, dtype=cp.int32)

    num_threads = 1024
    num_blocks = math.ceil(num_rows, num_threads)

    # run kernel
    SPMV_kernel[num_blocks, num_threads](num_rows, num_cols, num_bands, j_min_i, A, x, y)

    # synchronize
    cp.cuda.Stream.null.synchronize()
    
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

