import torch
import numpy as np
from typing import Tuple

# my personal implementation of generations for the matrices, vectors etc.
import generations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_pre_smoother_steps = 1
num_post_smoother_steps = 1
tolerance = 0.0
max_iter = 50

def computeDot(x: torch.tensor, y: torch.tensor) -> float:
    return torch.dot(x, y)

def computeSymGS(nx: int, nz: int, ny: int,
                 A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor)-> int:
    
    n_rows = nx * ny * nz
    indices = A.indices()
    values = A.values()

    # forward pass
    for i in range(n_rows):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]

        diag = (nnz_values[nnz_cols == i]).item()

        sum = r[i].item()
        sum -= torch.dot(nnz_values, x[nnz_cols])

        sum += diag * x[i].item()
        x[i] = sum / diag
    

    # backward pass
    for i in range(n_rows-1, -1, -1):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]

        diag = (nnz_values[nnz_cols == i]).item()

        sum = r[i].item()

        sum -= torch.dot(nnz_values, x[nnz_cols])
        sum += diag * x[i].item()
        x[i] = sum / diag


    return 0

def computeSPMV(nx: int, nz: int, ny: int,
                A: torch.sparse.Tensor, x: torch.Tensor, y: torch.Tensor)-> int:
    
    n_rows = nx * ny * nz
    indices = A.indices()
    values = A.values()

    for i in range(n_rows):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]

        y[i] = torch.dot(nnz_values, x[nnz_cols])

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
    
    # so there is a lot of stuff going on with mgData in the original. But it's already preprocessed there
    # so we can just assume that it's already done here and passed as this random thing called AmgData

    # in contrast to the reference implementation we compute the coarsening inside this function

    x.zero_()

    ierr = 0

    if depth < 3:

        # first generate new matrix
        # since that might throw an error and we don't want to do computation in that case

        f2c_op, Ac, _ = generations.generate_coarse_problem(nx, ny, nz)

        nxc = nx // 2
        nyc = ny // 2
        nzc = nz // 2

        nc = nxc * nyc * nzc
        Axf = torch.zeros(nx * ny * nz, device=device, dtype=torch.float64)
        rc = torch.zeros(nc, device=device, dtype=torch.float64)
        xc = torch.zeros(nc, device=device, dtype=torch.float64)

        for i in range(num_pre_smoother_steps):
            ierr += computeSymGS(nx, ny, nz, A, r, x)
        if ierr != 0:
            return ierr
        
        # Axf is where it is stored
        ierr = computeSPMV(nx, ny, nz, A, x, Axf)
        if ierr != 0:
            return ierr

        # Perform restriction operation using simple injection
        ierr = computeRestriction(Axf, r, nc, f2c_op, rc)
        if ierr != 0:
            return ierr
        
        # check that these attributes are correct

        ierr = computeMG(nxc, nyc, nzc, Ac, rc, xc, depth + 1)
        if ierr != 0:
            return ierr
        ierr = computeProlongation(x, xc, f2c_op, nc)
        if ierr != 0:
            return ierr
        
        for i in range(num_post_smoother_steps):
            ierr += computeSymGS(nx, ny, nz, A, r, x)
        if ierr != 0:
            return ierr
        
    else:
        ierr = computeSymGS(nx, ny, nz, A, r, x)
        if (ierr!=0):
            return ierr;  
    
    return 0

def computeWAXPBY(a: float, x: torch.Tensor, b: float, y: torch.Tensor, w: torch.Tensor)-> int:
    # note that double can also be x or y!
    w.copy_(a * x + b * y)
    return 0

def computeCG_no_preconditioning(nx: int, ny: int, nz: int,
                                 A: torch.sparse.Tensor, y: torch.Tensor, x: torch.Tensor) -> int:
    
    norm_r = 0.0
    
    # r: residual vector
    r = torch.zeros_like(y)

    # z: preconditioned residual vector
    z = torch.zeros_like(y)

    # p is of length ncols, copy x to p for sparse MV operation
    p = x.clone()
    Ap = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)

    computeSPMV(nx, ny, nz, A, p, Ap)

    # r = b - Ap
    computeWAXPBY(1.0, y, -1.0, Ap, r)

    norm_r = torch.sqrt(computeDot(r, r))
    norm_0 = norm_r

    for i in range(1, max_iter+1):
        if norm_r / norm_0 <= tolerance:
            break

        # in this version of the function we skip the preconditioning
        z = r.clone()
        
        if (i == 1):
            # copy Mr to p
            p = z.clone()
            # rtz = r'*z
            rtz = computeDot(r, z)
        else:
            oldrtz = rtz
            # rtz = r'*z
            rtz = computeDot(r, z)
            beta = rtz / oldrtz
            # p = beta*p + z
            computeWAXPBY(1.0, z, beta, p, p)

        # 
        computeSPMV(nx, ny, nz, A, p, Ap)
        # alpha = p'*Ap
        pAp = computeDot(p, Ap)
        alpha = rtz / pAp
        # x = x + alpha*p
        computeWAXPBY(1.0, x, alpha, p, x)
        # r = r - alpha*Ap
        computeWAXPBY(1.0, r, -alpha, Ap, r)
        norm_r = computeDot(r, r)
        norm_r = torch.sqrt(norm_r)
    
    return 0    

def computeCG(nx: int, ny: int, nz: int,
              A: torch.sparse.Tensor, y: torch.Tensor, x: torch.Tensor) -> int:

    norm_r = 0.0

    # r: residual vector
    r = torch.zeros_like(y)

    # z: preconditioned residual vector
    z = torch.zeros_like(y)

    # p is of length ncols, copy x to p for sparse MV operation
    p = x.clone()
    Ap = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)

    computeSPMV(nx, ny, nz, A, p, Ap)

    # r = b - Ap
    computeWAXPBY(1.0, y, -1.0, Ap, r)

    norm_r = torch.sqrt(computeDot(r, r))
    norm_0 = norm_r


    for i in range(1, max_iter+1):
        if norm_r / norm_0 <= tolerance:
            break
        # we always want to do the preconditioning
        # we have a seperate function for no preconditioning
        computeMG(nx, ny, nz, A, y, z, 0)
        
        if (i == 1):
            # copy Mr to p
            p = z.clone()
            # rtz = r'*z
            rtz = computeDot(r, z)
        else:
            oldrtz = rtz
            # rtz = r'*z
            rtz = computeDot(r, z)
            beta = rtz / oldrtz
            # p = beta*p + z
            computeWAXPBY(1.0, z, beta, p, p)

        # 
        computeSPMV(nx, ny, nz, A, p, Ap)
        # alpha = p'*Ap
        pAp = computeDot(p, Ap)
        alpha = rtz / pAp
        # x = x + alpha*p
        computeWAXPBY(1.0, x, alpha, p, x)
        # r = r - alpha*Ap
        computeWAXPBY(1.0, r, -alpha, Ap, r)
        norm_r = computeDot(r, r)
        norm_r = torch.sqrt(norm_r)
    
    return 0    

#################################################################################################################
# this is only a test thingy

num = 8


A,y = generations.generate_torch_coo_problem(num,num,num)
x = torch.zeros(num*num*num, device=device, dtype=torch.float64)

computeSPMV(num, num, num, A, y, x)

# computeMG(num, num, num, A,y,x,0)
# computeCG(num, num, num, A, y, x)
# print("computed CG")
#################################################################################################################

