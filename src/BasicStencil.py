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
debug = False # this will also skip the preconditioning in the CG function


#################################################################################################################
# Static util datastructure for the banded matrix
#################################################################################################################
# define the offsets for the bands
offsets_to_band_index = {
        (-1, -1, -1): 0, (0, -1, -1): 1, (1, -1, -1): 2,
        (-1, 0, -1): 3, (0, 0, -1): 4, (1, 0, -1): 5,
        (-1, 1, -1): 6, (0, 1, -1): 7, (1, 1, -1): 8,
        (-1, -1, 0): 9, (0, -1, 0): 10, (1, -1, 0): 11,
        (-1, 0, 0): 12, (0, 0, 0): 13, (1, 0, 0): 14,
        (-1, 1, 0): 15, (0, 1, 0): 16, (1, 1, 0): 17,
        (-1, -1, 1): 18, (0, -1, 1): 19, (1, -1, 1): 20,
        (-1, 0, 1): 21, (0, 0, 1): 22, (1, 0, 1): 23,
        (-1, 1, 1): 24, (0, 1, 1): 25, (1, 1, 1): 26
    }

sx_sy_sz_offsets = list(offsets_to_band_index.keys())
#################################################################################################################

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
    band_matrix = torch.zeros(nx*ny*nz, 27, device=device, dtype=torch.float64)

    indices = A.indices()
    values = A.values()

    offsets_in_matrix_sx_sy_sz = [(sx + nx*sy + nx * ny* sz, sx, sy, sz) for sx, sy, sz in sx_sy_sz_offsets]

    # we iterate over the matrix
    for elem in range(indices.size(1)):
        i = indices[0, elem].item()
        j = indices[1, elem].item()
        value = values[elem].item()

        # find the right band and update the banded matrix
        for band_offset, sx, sy, sz in offsets_in_matrix_sx_sy_sz:
            if j == i + band_offset:
                band_index = offsets_to_band_index[(sx, sy, sz)]
                band_matrix[i, band_index] = value
                break
    
    return band_matrix

def computeDot(x: torch.tensor, y: torch.tensor) -> float:
    print(f"WARNING: computeDot not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeDot(x, y)
    
def computeSymGS(nx: int, nz: int, ny: int,
                 A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor)-> int:
    print(f"WARNING: computeSymGS not implemented for {version_name}, using BaseTorch implementation")
    return BaseTorch.computeSymGS(nx, nz, ny, A, r, x)

def computeSPMV(nx: int, nz: int, ny: int,
                A: torch.Tensor, x: torch.Tensor, y: torch.Tensor)-> int:
    
    """
    computes Ax = y
    """
    
    offsets_in_matrix_sx_sy_sz = [(sx + nx*sy + nx * ny* sz, sx, sy, sz) for sx, sy, sz in sx_sy_sz_offsets]
    
    # note: we added zero-padding to the matrix, therefore we don't have to check for the boundaries
    # while that is true for the matrix, we still have to check for the boundaries of the vector

    # set y to zero
    y.zero_()

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                for j_min_i, sx, sy, sz in offsets_in_matrix_sx_sy_sz:
                    j = i + j_min_i
                    # this is to make sure we don't go out of bounds on the vectors x & y
                    if j > 0 and j < nx*ny*nz:
                        y[i] += A[i, offsets_to_band_index[(sx, sy, sz)]] * x[j]

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



#################################################################################################################
# this is only a test thingy

# num = 2



# A,y = generations.generate_torch_coo_problem(num,num,num)
# x = torch.zeros(num*num*num, device=device, dtype=torch.float64)
# print("generated problem, starting conversion")
# dense_A = convert_A_to_Band_matrix(num,num,num,A)
# print("conversion ended starting computation")
# computeSPMV(num,num,num,dense_A,y,x)
# print("computed SPMV")

# computeMG(num, num, num, A,y,x,0)
# computeCG(num, num, num, A, y, x)
# print("computed CG")
#################################################################################################################

