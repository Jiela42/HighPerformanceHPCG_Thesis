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
                 A: torch.sparse.Tensor, r: np.array, x: np.array)-> int:
    
    

    return 1

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

def test_AMGX():
    import scipy.sparse as sparse
    import scipy.sparse.linalg as splinalg

    def is_positive_definite(sparse_matrix):
        # Compute the smallest eigenvalue
        smallest_eigenvalue = splinalg.eigsh(sparse_matrix, k=1, which='SA', return_eigenvectors=False)[0]
        return smallest_eigenvalue > 0 


    print("PyAMGX version quicktest")

    pyamgx.initialize()

    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict({
    "config_version": 2,
            "determinism_flag": 1,
            "exception_handling" : 1,
            # "max_iters": 1,
            "solver": {
                # "monitor_residual": 1,
                "solver": "MULTICOLOR_GS",
                # "ColoringType": "halo_coloring",
                "symmetric_GS": 1, 
                "max_iters": 1,
                "solver_verbose":2,
                "relaxation_factor": 1,
                # "obtain_timings": 1,
                # "convergence": "RELATIVE_INI_CORE",
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
    # meta_data , A_csr, y = generations.generate_cupy_csr_problem(2,2,2)
    # meta_data, A_csr, y = generations.generate_cupy_csr_problem(4,4,4)
    # meta_data , A_csr, y = generations.generate_cupy_csr_problem(8,8,8)
    # meta_data, A_csr, y = generations.generate_cupy_csr_problem(16,16,16)
    # meta_data , A_csr, y = generations.generate_cupy_csr_problem(32,32,32)

    meta_data , A_csr, y = generations.generate_cupy_csr_problem(3,4,5)
    # meta_data , A_csr, y = generations.generate_cupy_csr_problem(4,3,5)

    sol_torch = torch.zeros(meta_data.num_rows, device=device, dtype=torch.float64)
    sol = np.zeros(meta_data.num_rows, dtype=np.float64)

    A_csr_scipy = sparse.csr_matrix((A_csr.data.get(), A_csr.indices.get(), A_csr.indptr.get()), shape=A_csr.shape)
    y = y.get()

    #################################################################################################################
    # get torch stuff
    # Convert the CuPy CSR matrix to a COO format
    A_coo = A_csr.tocoo()

    # Extract the COO format data
    row = A_coo.row
    col = A_coo.col
    data = A_coo.data

    # Convert these arrays to PyTorch tensors
    row_torch = torch.tensor(row, dtype=torch.int64)
    col_torch = torch.tensor(col, dtype=torch.int64)
    data_torch = torch.tensor(data, dtype=torch.float64)

    # Create a PyTorch sparse COO tensor
    indices = torch.stack([row_torch, col_torch])
    A_torch = torch.sparse_coo_tensor(indices, data_torch, size=A_coo.shape, device=device)
    A_torch = A_torch.coalesce()

    y_torch = torch.tensor(y, dtype=torch.float64, device=device)
    sol_torch = torch.zeros(meta_data.num_rows, device=device, dtype=torch.float64)

    BaseTorch.computeSymGS(meta_data.nx, meta_data.ny, meta_data.nz, A_torch, y_torch, sol_torch)

    if is_positive_definite(A_csr_scipy):
        print("A is positive definite")
    else:
        print("A is not positive definite")

    fun_matrix = np.array([
        [1, 2, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 1, 2]
    ], dtype=np.float64)

    fun_matrix_scipy = sparse.csr_matrix(fun_matrix)
    if is_positive_definite(fun_matrix_scipy):
        print("fun_matrix is positive definite")
    else:
        print("fun_matrix is not positive definite")

    #################################################################################################################

    A.upload_CSR(A_csr_scipy)
    b.upload(y)
    x.upload(sol)

    # Setup and solve system:
    solver.setup(A)
    solver.solve(b, x)

    # Download solution
    x.download(sol)
    print("pyamgx solution: ", sol[:10])
    print("torch solution: ", sol_torch[:10])
    # print("scipy solution: ", splinalg.spsolve(A_csr_scipy, y)[:10])


    # Clean up:
    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    pyamgx.finalize()
#################################################################################################################

# test_AMGX()