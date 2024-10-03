import torch
import numpy as np
from typing import Tuple

# my personal implementation of generations for the matrices, vectors etc.
import generations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_pre_smoother_steps = 1
num_post_smoother_steps = 1
tolerance = 0.0
n_iter = 50


def computeSPMV_stencil(nx: int, ny: int, nz: int, y: torch.tensor, x: torch.tensor) -> int:

    # iterate over the rows of the matrix
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                col_idx_i = []
                # iterate over the neighbours
                for sz in range(-1, 2):
                    if iz+sz > -1 and iz+sz < nz:
                        for sy in range(-1, 2):
                            if iy+sy > -1 and iy+sy < ny:
                                for sx in range(-1, 2):
                                    if ix+sx > -1 and ix+sx < nx:
                                        j = ix+sx + nx*(iy+sy) + nx*ny*(iz+sz)
                                        if i == j:
                                            # Ax = y
                                            y[i] += 26.0 * x[j]
                                        else:
                                            y[i] -= 1.0 * x[j]
    return 0

def computeDot(x: torch.tensor, y: torch.tensor) -> float:
    return torch.dot(x, y)

def computeSymGS(nx: int, nz: int, ny: int,
                 A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor)-> int:
    
    n_rows = nx * ny * nz
    indices = A.indices()
    values = A.values()

    # print("n_rows: ", n_rows)

    # # make x all zeros
    # x[:] = 0.0

    # CHECK x vs r 

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
    
    # print(f"BaseTorch x between passes: {x}")

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

    # print(f"In the computeMG function, depth: {depth} - x dtype: {x.dtype}")
    ierr = 0

    if depth < 3:
        # print("In the if clause, depth: ", depth)

        # first generate new matrix
        # since that might throw an error and we don't want to do computation in that case

        # print("Starting to generate coarse problem, nx: ", nx, " ny: ", ny, " nz: ", nz)
        # f2c_op, Ac, _ = generations.generate_coarse_problem(nx, ny, nz)
        f2c_op, Ac, _ = generations.generate_coarse_problem(nx, ny, nz)
        # print("End of generating coarse problem")

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

def computeWAXPBY(a: torch.float64, x: torch.Tensor, b: torch.float64, y: torch.Tensor, w: torch.Tensor)-> int:
    w = a * x + b * y
    return 0

def computeCG(nx: int, ny: int, nz: int) -> int:

    norm_err = 0.0

    A,y = generations.generate_torch_coo_problem(nx,ny,nz)
    x = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)


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


    for i in range(n_iter) and norm_err > tolerance:
        computeMG(nx, ny, nz, A, y, z, 0)
    
    return 0

"""
// p is of length ncols, copy x to p for sparse MV operation
  CopyVector(x, p);
  TICK(); ComputeSPMV_ref(A, p, Ap);  TOCK(t3); // Ap = A*p
  TICK(); ComputeWAXPBY_ref(nrow, 1.0, b, -1.0, Ap, r); TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); ComputeDotProduct_ref(nrow, r, r, normr, t4);  TOCK(t1);
  normr = sqrt(normr);
#ifdef HPCG_DEBUG
  if (A.geom->rank==0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations

  for (int k=1; k<=max_iter && normr/normr0 > tolerance; k++ ) {
    TICK();
    if (doPreconditioning)
      ComputeMG_ref(A, r, z); // Apply preconditioner
    else
      ComputeWAXPBY_ref(nrow, 1.0, r, 0.0, r, z); // copy r to z (no preconditioning)
    TOCK(t5); // Preconditioner apply time

    if (k == 1) {
      CopyVector(z, p); TOCK(t2); // Copy Mr to p
      TICK(); ComputeDotProduct_ref(nrow, r, z, rtz, t4); TOCK(t1); // rtz = r'*z
    } else {
      oldrtz = rtz;
      TICK(); ComputeDotProduct_ref(nrow, r, z, rtz, t4); TOCK(t1); // rtz = r'*z
      beta = rtz/oldrtz;
      TICK(); ComputeWAXPBY_ref(nrow, 1.0, z, beta, p, p);  TOCK(t2); // p = beta*p + z
    }

    TICK(); ComputeSPMV_ref(A, p, Ap); TOCK(t3); // Ap = A*p
    TICK(); ComputeDotProduct_ref(nrow, p, Ap, pAp, t4); TOCK(t1); // alpha = p'*Ap
    alpha = rtz/pAp;
    TICK(); ComputeWAXPBY_ref(nrow, 1.0, x, alpha, p, x);// x = x + alpha*p
            ComputeWAXPBY_ref(nrow, 1.0, r, -alpha, Ap, r);  TOCK(t2);// r = r - alpha*Ap
    TICK(); ComputeDotProduct_ref(nrow, r, r, normr, t4); TOCK(t1);
    normr = sqrt(normr);
"""
    



#################################################################################################################
# this is only a test thingy

num = 16


# A,y = generations.generate_torch_coo_problem(num,num,num)
# x = torch.zeros(num*num*num, device=device, dtype=torch.float64)

# computeMG(num, num, num, A,y,x,0)
# computeCG(num, num, num)
#################################################################################################################

