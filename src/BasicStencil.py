import BaseTorch
import torch

import torch
import numpy as np
from typing import Tuple

# my personal implementation of generations for the matrices, vectors etc.
import generations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

version_name = "BasicStencil"

num_pre_smoother_steps = 1
num_post_smoother_steps = 1
tolerance = 0.0
max_iter = 50
debug = True # this will also skip the preconditioning in the CG function

def convert_A_to_Band_matrix(nx: int, ny: int, nz: int, A: torch.sparse.Tensor) -> torch.Tensor:
    """
    Converts a sparse matrix to a band matrix.

    Parameters:
    nx [in] (int): Number of grid points in the x-direction.
    ny [in] (int): Number of grid points in the y-direction.
    nz [in] (int): Number of grid points in the z-direction.
    A [in] (torch.sparse.Tensor): The sparse matrix.

    Returns:
    torch.Tensor: The band matrix containing 9 bands.
                band_matrix[0] is the first row
    """
    band_matrix = torch.zeros(nx*ny*nz, 9, device=device, dtype=torch.float64)

    indices = A.indices()
    values = A.values()

    # we need to figure out how to map from A to the band matrix based on nx, ny, nz


    return band_matrix


def computeDot(x: torch.tensor, y: torch.tensor) -> float:
    print(f"WARNING: computeDot not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeDot(x, y)
    
def computeSymGS(nx: int, nz: int, ny: int,
                 A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor)-> int:
    print(f"WARNING: computeSymGS not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeSymGS(nx, nz, ny, A, r, x)

def computeSPMV(nx: int, nz: int, ny: int,
                A: torch.sparse.Tensor, x: torch.Tensor, y: torch.Tensor)-> int:
    
    # inner nodes
    for ix in range(1, nx-1):
        for iy in range(1, ny-1):
            for iz in range(1, nz-1):
                i = ix + nx*iy + nx*ny*iz
                for sz in range(-1, 2):
                    for sy in range(-1, 2):
                        for sx in range(-1, 2):
                            j = ix+sx + nx*(iy+sy) + nx*ny*(iz+sz)
                            y[i] += A[i,j] * x[j]
                
    
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
#################################################################################################################

