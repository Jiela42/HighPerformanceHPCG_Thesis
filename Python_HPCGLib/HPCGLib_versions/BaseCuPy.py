import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


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
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import getSymGS_rrNorm, getSymGS_rrNorm_zero_based

version_name = "BaseCuPy"

num_pre_smoother_steps = 1
num_post_smoother_steps = 1
tolerance = 0.0
max_iter = 50
debug = True # this will also skip the preconditioning in the CG function

# lsmr for size 2x2x2 converged at 2 iterations
# lsmr for size 4x4x4 converged at 5 iterations
# lsmr for size 8x8x8 converged at 6 iterations
# lsmr for size 16x16x16 converged at 8 iterations
# lsmr for size 24x24x24 converged at 8 iterations
# lsmr for size 32x32x32 converged at 8 iterations
# lsmr for size 64x64x64 converged at 7 iterations
# lsmr for size 128x64x64 converged at 7 iterations
# lsmr for size 128x128x64 converged at 7 iterations
# lsmr for size 128x128x128 converged at 7 iterations

num_its_lsrm = {
    (2,2,2): 2,
    (4,4,4): 5,
    (8,8,8): 6,
    (16,16,16): 8,
    (24,24,24): 8,
    (32,32,32): 8,
    (64,64,64): 7,
    (128,64,64): 7,
    (128,128,64): 7,
    (128,128,128): 7,
}

num_its_minres = {
    (2,2,2): 2,
    (4,4,4): 4,
    (8,8,8): 4,
    (16,16,16): 4,
    (24,24,24): 4,
    (32,32,32): 4,
    (64,64,64): 4,
    (128,64,64): 4,
    (128,128,64): 4,
    (128,128,128): 4,
}

num_its_lsrm_zeroBased = {
    (2,2,2): 2,
    (4,4,4): 3,
    (8,8,8): 3,
    (16,16,16): 4,
    # (24,24,24): 8,
    (32,32,32): 5,
    (64,64,64): 6,
    (128,64,64): 6,
    (128,128,64): 6,
    (128,128,128): 5,
}

# minres for size 2x2x2 (zerobased) converged at 1 iterations
# minres for size 4x4x4 (zerobased) converged at 3 iterations
# minres for size 8x8x8 (zerobased) converged at 4 iterations
# minres for size 16x16x16 (zerobased) converged at 4 iterations
# minres for size 32x32x32 (zerobased) converged at 4 iterations
# minres for size 64x64x64 (zerobased) converged at 4 iterations
# minres for size 128x64x64 (zerobased) converged at 4 iterations
# minres for size 128x128x64 (zerobased) converged at 4 iterations

num_its_minres_zeroBased = {
    (2,2,2): 1,
    (4,4,4): 2,
    (8,8,8): 3,
    (16,16,16): 3,
    (32,32,32): 3,
    (64,64,64): 3,
    (128,64,64): 3,
    (128,128,64): 3,
    (128,128,128): 3,
}

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
    
    solution = cupyx.scipy.sparse.linalg.lsmr(A, b=y, x0=x, maxiter=1)
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


def get_num_its_lsmr(nx, ny, nz):

    # print(f"Getting the number of iterations for {nx}x{ny}x{nz}")

    # Generate the matrix
    A_CSR, A_cupy, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    x = np.zeros_like(y)
    original_x = x.copy()
    rr_norm_threshold = getSymGS_rrNorm_zero_based(nx, ny, nz)

    # print(f"rr_norm_threshold: {rr_norm_threshold}")

    # for each of the solvers, run the solver and get the number of iterations
    solution = cupyx.scipy.sparse.linalg.lsmr(A_cupy, b=y, x0=x)
    rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)
    # print(f"LSMR: {solution[2]} iterations has rr_norm: {rr_norm:.10f}")

    # print("lsmr solution: ", solution[0])

    # print("L2 Norm of ax-y: ", np.linalg.norm(y - A_cupy.dot(solution[0])))
    # print("L2 Norm of y: ", np.linalg.norm(y))

    # zero out the solution
    x = original_x.copy()

    while rr_norm < rr_norm_threshold:
        solution = cupyx.scipy.sparse.linalg.lsmr(A_cupy, b=y, x0=x, maxiter=solution[2]-1)
        rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)
        # print(f"LSMR: {solution[2]} iterations has rr_norm: {rr_norm:.10f}")
        x = original_x.copy()

    print(f"lsmr for size {nx}x{ny}x{nz} (zerobased) converged at {solution[2] + 1} iterations", flush=True)

def get_num_its_minres(nx, ny, nz):

    # print(f"Getting the number of iterations for {nx}x{ny}x{nz}")

    # Generate the matrix
    A_CSR, A_cupy, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    x = cp.random.rand(y.shape[0])
    original_x = x.copy()
    rr_norm_threshold = getSymGS_rrNorm(nx, ny, nz)
    # print(f"rr_norm_threshold: {rr_norm_threshold}")

    # somehow this one doesn't return the number of iterations
    # so we find the number of iterations the old fashioned way

    num_iterations = 1


    x = original_x.copy()

    rr_norm = 10

    while rr_norm > rr_norm_threshold:
        solution = cupyx.scipy.sparse.linalg.minres(A_cupy, b=y, x0=x, maxiter=num_iterations)
        rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)
        # print(f"MINRES: {num_iterations} iterations has rr_norm: {rr_norm:.10f}")
        x = original_x.copy()
        num_iterations += 1

    print(f"minres for size {nx}x{ny}x{nz} (randomized) converged at {solution[1] + 1} iterations", flush=True)


def check_num_its(nx, ny, nz):

    A_CSR, A_cupy, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    x_zero = cp.zeros_like(y)
    x_randomized = cp.random.rand(y.shape[0])
    x_zero_original = x_zero.copy()
    x_randomized_original = x_randomized.copy()

    rr_norm_threshold_randomized = getSymGS_rrNorm(nx, ny, nz)
    rr_norm_threshold_zeroBased = getSymGS_rrNorm_zero_based(nx, ny, nz)

    # now we run both solvers for both versions and see if the number of iterations is enough to go below the threshold

    check_randomized = False

    fail_test_offset = 0

    failure_ctr_lsmr_zeroBased = 0
    failure_ctr_minres_zeroBased = 0
    failure_ctr_lsmr_randomized = 0
    failure_ctr_minres_randomized = 0

    for i in range (20):

        x_zero = x_randomized_original.copy()

        if check_randomized:
            x_randomized = cp.random.rand(y.shape[0])
            current_num_its_lsmr = num_its_lsrm.get((nx, ny, nz)) + fail_test_offset
            solution = cupyx.scipy.sparse.linalg.lsmr(A_cupy, b=y, x0=x_randomized, maxiter=current_num_its_lsmr)
            rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)

            if rr_norm > rr_norm_threshold_randomized:
                failure_ctr_lsmr_randomized += 1
                # print(f"lsmr for size {nx}x{ny}x{nz} (randomized) did not converge at {current_num_its_lsmr} iterations", flush=True)

        
        current_num_its_lsmr_zeroBased = num_its_lsrm_zeroBased.get((nx, ny, nz)) + fail_test_offset
        solution = cupyx.scipy.sparse.linalg.lsmr(A_cupy, b=y, x0=x_zero, maxiter=current_num_its_lsmr_zeroBased)
        rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)

        if rr_norm > rr_norm_threshold_zeroBased:
            failure_ctr_lsmr_zeroBased += 1
            # print(f"lsmr for size {nx}x{ny}x{nz} (zerobased): rr_norm is {rr_norm}, but should be below {rr_norm_threshold_zeroBased}", flush=True)
            # print(f"lsmr for size {nx}x{ny}x{nz} (zerobased) did not converge at {current_num_its_lsmr_zeroBased} iterations", flush=True)
        
        
        x_zero = x_zero_original.copy()

        if check_randomized:
            x_randomized = cp.random.rand(y.shape[0])

            current_num_its_minres = num_its_minres.get((nx, ny, nz)) + fail_test_offset
            solution = cupyx.scipy.sparse.linalg.minres(A_cupy, b=y, x0=x_randomized, maxiter=current_num_its_minres)
            rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)

            if rr_norm > rr_norm_threshold_randomized:
                failure_ctr_minres_randomized += 1
                # print(f"minres for size {nx}x{ny}x{nz} (randomized): rr_norm is {rr_norm}, but should be below {rr_norm_threshold_randomized}", flush=True)
                # print(f"minres for size {nx}x{ny}x{nz} (randomized) did not converge at {current_num_its_minres} iterations", flush=True)
        
        current_num_its_minres_zeroBased = num_its_minres_zeroBased.get((nx, ny, nz)) + fail_test_offset
        solution = cupyx.scipy.sparse.linalg.minres(A_cupy, b=y, x0=x_zero, maxiter=current_num_its_minres_zeroBased)
        rr_norm = np.linalg.norm(y - A_cupy.dot(solution[0]))/np.linalg.norm(y)

        if rr_norm > rr_norm_threshold_zeroBased:
            failure_ctr_minres_zeroBased += 1
            # print(f"minres for size {nx}x{ny}x{nz} (zerobased) did not converge at {current_num_its_minres_zeroBased} iterations", flush=True)

    print(f"lsmr for size {nx}x{ny}x{nz} (zerobased) failed to converge {failure_ctr_lsmr_zeroBased} times out of 20", flush=True)
    print(f"minres for size {nx}x{ny}x{nz} (zerobased) failed to converge {failure_ctr_minres_zeroBased} times out of 20", flush=True)

    if check_randomized:
        print(f"lsmr for size {nx}x{ny}x{nz} (randomized) failed to converge {failure_ctr_lsmr_randomized} times out of 20", flush=True)
        print(f"minres for size {nx}x{ny}x{nz} (randomized) failed to converge {failure_ctr_minres_randomized} times out of 20", flush=True)


if __name__ == "__main__":
    # This portion helped me figure out exactly how many iterations I need for each of the solvers
    
    
    # get_num_its_lsmr(2,2,2)
    # get_num_its_lsmr(4,4,4)
    # get_num_its_lsmr(8,8,8)
    # get_num_its_lsmr(16,16,16)
    # # get_num_its_lsmr(24,24,24)
    # get_num_its_lsmr(32,32,32)
    # get_num_its_lsmr(64,64,64)
    # get_num_its_lsmr(128,64,64)
    # get_num_its_lsmr(128,128,64)
    # get_num_its_lsmr(128,128,128)


    # get_num_its_minres(2,2,2)
    # get_num_its_minres(4,4,4)
    # get_num_its_minres(8,8,8)
    # get_num_its_minres(16,16,16)
    # get_num_its_minres(24,24,24)
    # get_num_its_minres(32,32,32)
    # get_num_its_minres(64,64,64)
    # get_num_its_minres(128,64,64)
    # get_num_its_minres(128,128,64)
    # get_num_its_minres(128,128,128)
    

    # check_num_its(2,2,2)
    # check_num_its(4,4,4)
    # check_num_its(8,8,8)
    # check_num_its(16,16,16)
    # check_num_its(24,24,24)
    # check_num_its(32,32,32)
    # check_num_its(64,64,64)
    # check_num_its(128,64,64)
    check_num_its(128,128,64)
    # check_num_its(128,128,128)

    # A_CSR, A_cupy, y = generations.generate_cupy_csr_problem(4, 4, 4)
    # x = np.zeros_like(y)

    # computeSymGS_lsmr(A_CSR, A_cupy, x, y)
    # print(f"lsmr solution: {x}")

    # rr_norm = np.linalg.norm(y - A_cupy.dot(x))/np.linalg.norm(y)

    # get_num_its(8, 8, 8)


