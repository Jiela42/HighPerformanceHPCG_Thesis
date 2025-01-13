import torch

import torch
import numpy as np
from typing import Tuple

import pyamgx

import sys
import os


print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# my personal implementation of generations for the matrices, vectors etc.
import HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.generations as generations


import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

version_name = "pyamgx"

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

def computeSPMV(nx: int, nz: int, ny: int,
                A: torch.sparse.Tensor, x: torch.Tensor, y: torch.Tensor)-> int:
    
    print(f"WARNING: computeSPMV not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeSPMV(nx, nz, ny, A, x, y)

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



#################################################################################################################
# this is only a test thingy

# num = 8


# A,y = generations.generate_torch_coo_problem(num,num,num)
# x = torch.zeros(num*num*num, device=device, dtype=torch.float64)

# computeMG(num, num, num, A,y,x,0)
# computeCG(num, num, num, A, y, x)
# print("computed CG")

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

print("PyAMGX version quicktest")

pyamgx.initialize()

# Initialize config and resources:
cfg = pyamgx.Config().create_from_dict({
   "config_version": 2,
        "determinism_flag": 1,
        "exception_handling" : 1,
        "solver": {
            "monitor_residual": 1,
            "solver": "BICGSTAB",
            "convergence": "RELATIVE_INI_CORE",
            "preconditioner": {
                "solver": "NOSOLVER"
        }
    }
})

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

# Upload system:

M = sparse.csr_matrix(np.random.rand(5, 5))
rhs = np.random.rand(5)
sol = np.zeros(5, dtype=np.float64)

A.upload_CSR(M)
b.upload(rhs)
x.upload(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)

# Download solution
x.download(sol)
print("pyamgx solution: ", sol)
print("scipy solution: ", splinalg.spsolve(M, rhs))

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()
#################################################################################################################

