import torch.tensor
import torch
import triton
import triton.language as tl
import numpy as np
from typing import Tuple
from ctypes import POINTER, c_float

import BaseTorch

# my personal implementation of generations for the matrices, vectors etc.
import generations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

version_name = "BaseTriton"

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

@triton.jit
def computeSPMV(nx: int, nz : int, ny : int,
                # A_row_ptr: torch.tensor, A_colIdx_ptr: torch.tensor, A_val_ptr: torch.tensor,
                # x_ptr: torch.tensor, y_ptr: torch.tensor,
                # out: torch.tensor,
                A_row_ptr, A_colIdx_ptr, A_val_ptr,
                x_ptr , y_ptr,
                out,
                BLOCK_SIZE: tl.constexpr
                ):
    
    tl.store(out, 1)
    """
    Computes Ax and stores it in y.
    """

    n_rows = nx * ny * nz

    # get the thread id
    tid0 = tl.program_id(0)
    tid1 = tl.program_id(1)
    tid2 = tl.program_id(2)

    # each computation unit will compute Block_size many elements of the result vector
    block_start = tid0 * BLOCK_SIZE
    row_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < n_rows

    # # load the data
    A_row_idx_start = tl.load(A_row_ptr + row_offsets, mask=row_mask)
    A_row_idx_end = tl.load(A_row_ptr + row_offsets + 1, mask=row_mask)
    # maybe loads and stores are asynchronous? I don't know. fuck.

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float64)

    start = A_row_idx_start[0]

    # for row in range(0,BLOCK_SIZE):
        # if tid0 == 0 and (tid1 == 0 and tid2 == 0):
        #     print("rowmask type: ", (row_mask[row]))
        # print(A_row_idx_start)


        # if row_mask[row]:
            # print("row: ", row)
            # get the row start and end indecies into the column index and value arrays
        # start = A_row_idx_start[row]
    #     end = A_row_idx_end[row]

    #     for idx in range(start, end):
    #             col = tl.load(A_colIdx_ptr + idx)
    #             val = tl.load(A_val_ptr + idx)
    #             result[row] += val * tl.load(x_ptr + col)
    
    # tl.store(y_ptr + row_offsets, result, mask=row_mask)

    tl.store(out, 0)



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

num = 4


A_rowIdx, A_colIdx, A_vals, y = generations.generate_torch_csr_problem(num,num,num)
x = torch.zeros(num*num*num, device=device, dtype=torch.float64)

print("generated problem, starting Computation", flush=True)

num_rows = num*num*num
error_out = torch.tensor(0, device=device, dtype=torch.int32)
max_num_threads = 1024


grid = lambda meta: (triton.cdiv(num_rows, meta['BLOCK_SIZE']),)
# grid = (num_rows, 32, 32)

# print(f"device A_rowIdx: {A_rowIdx.device}")
# print(f"device A_colIdx: {A_colIdx.device}")
# print(f"device A_vals: {A_vals.device}")
# print(f"device x: {x.device}")
# print(f"device y: {y.device}")

# Each thread will handle blocksize many rows, to adjust the amount of threads, might be a good idea
computeSPMV[grid](num, num, num, A_rowIdx, A_colIdx, A_vals, x, y, error_out, BLOCK_SIZE=32)


# computeMG(num, num, num, A,y,x,0)
# computeCG(num, num, num, A, y, x)
# print("computed CG")
#################################################################################################################

