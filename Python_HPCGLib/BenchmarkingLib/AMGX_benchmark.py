# this file is different, because it benchmarks AMGX, which is not part of the HPCG_versions
# the reason for this is that AMGX takes different inputs
# and the way it's setup we don't want to mess with the scope of initialization

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import ault_node, num_bench_iterations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import getSymGS_rrNorm_zero_based, num_its_zerobased_AMGX
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.arbitrary_stencil import Shape, generate_stencil_matrix

import pyamgx
import scipy.sparse as sparse
import numpy as np
import cupy as cp

def run_AMGX_SymGS(A_csr_scipy, x_np, y, num_iterations):

    # in the beginning x is going to be all zeros.
    # remember to zero it out between runs

    pyamgx.initialize()
    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict({
    "config_version": 2,
            "determinism_flag": 0,
            "exception_handling" : 1,
            "print_coloring_info": 1,
            # "max_iters": 1,
            "solver": {
                # "monitor_residual": 1,
                "solver": "MULTICOLOR_GS",
                "symmetric_GS": 1, 
                "max_iters": num_iterations,
                # "solver_verbose":1,
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

    A.upload_CSR(A_csr_scipy)
    b.upload(y)
    x.upload(x_np)
    
    # Setup and solve system:
    print("setup and solve system")
    x.upload(np.zeros_like(x_np))
    solver.setup(A)
    solver.solve(b, x)
        # zero out x

    x.download(x_np)

    # print(f"A_csr_scipy shape: {A_csr_scipy.shape}")
    # print(f"x_np shape: {x_np.shape}")
    # print(f"y shape: {y.shape}")

    # print(f"Solution: {x_np[:10]}")

    # we compute a norm
    # Ax - b
    # norm = np.linalg.norm(cp.array(A_csr_scipy @ x_np) - cp.array(y))
    # print(f"Norm: {norm}")
    # Download solution

    # Clean up
    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    pyamgx.finalize()
    return

def bench_AMGX_SymGS(nx, ny, nz, A_csr_scipy, x_np, y, timer):

    # in the beginning x is going to be all zeros.
    # remember to zero it out between runs

    num_iterations = num_its_zerobased_AMGX[(nx,ny,nz)]

    pyamgx.initialize()
    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict({
    "config_version": 2,
            "determinism_flag": 1,
            "exception_handling" : 1,
            "print_coloring_info": 1,
            # "max_iters": 1,
            "solver": {
                # "monitor_residual": 1,
                "solver": "MULTICOLOR_GS",
                "symmetric_GS": 1, 
                "max_iters": num_iterations,
                # "solver_verbose":1,
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

    A.upload_CSR(A_csr_scipy)
    b.upload(y)
    x.upload(x_np)
    
    # Setup and solve system:
    for i in range (1):
        print("setup and solve system")
        x.upload(np.zeros_like(x_np))
        timer.start_timer()
        solver.setup(A)
        solver.solve(b, x)
        timer.stop_timer("computeSymGS")
        # zero out x

    x.download(x_np)

    # print(f"Solution: {x_np[:10]}")

    # we compute a norm
    # Ax - b
    norm = np.linalg.norm(cp.array(A_csr_scipy @ x_np) - cp.array(y))
    # print(f"Norm: {norm}")
    # Download solution

    timer.update_additional_info(timer.get_additional_info() + f"L2 Norm: {norm}")


    # Clean up
    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    pyamgx.finalize()
    return

def run_AMGX_CG(A_csr_scipy, x_np, y):
     # in the beginning x is going to be all zeros.
    # remember to zero it out between runs

    initial_norm = np.linalg.norm(cp.array(A_csr_scipy @ x_np) - cp.array(y))

    print(f"initial guess: {x_np[:10]}")

    pyamgx.initialize()
    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict({
    "config_version": 2,
            "determinism_flag": 1,
            "exception_handling" : 1,
            "print_coloring_info": 1,
            "solver": {
                "monitor_residual": 1,
                # "print_solve_stats": 1,
                "convergence_analysis": 1,
                "solver": "PCG",
                "solver_verbose":1,
                "relaxation_factor": 1,
                "obtain_timings": 1,
                "convergence": "RELATIVE_INI",
                "tolerance": 1e-12,
                "norm": "L2",
                "max_iters": 50,
                "preconditioner": {
                    "solver": "AMG",
                    "algorithm": "CLASSICAL",
                    "smoother": {
                        "solver": "MULTICOLOR_GS",
                        "symmetric_GS": 1, 
                        "max_iters": 1,
                    },
                    "relaxation_factor": 1,
                    "presweeps": 1,
                    "postsweeps": 1,
                    "max_levels": 3
                    # "scope": "amg"
                    
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

    A.upload_CSR(A_csr_scipy)
    b.upload(y)
    x.upload(x_np)
    
    # Setup and solve system:
    print("setup and solve system")
    x.upload(np.zeros_like(x_np))
    solver.setup(A)
    solver.solve(b, x)
        # zero out x

    x.download(x_np)

    # print(f"A_csr_scipy shape: {A_csr_scipy.shape}")
    # print(f"x_np shape: {x_np.shape}")
    # print(f"y shape: {y.shape}")

    # print(f"Solution: {x_np[:10]}")

    # we compute a norm
    # Ax - b
    norm = np.linalg.norm(cp.array(A_csr_scipy @ x_np) - cp.array(y))

    relative_norm = norm / initial_norm
    print(f"Relative norm: {relative_norm}")
    print(f"solution: {x_np[:10]}")
    # print(f"Norm: {norm}")
    # Download solution

    # Clean up
    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    pyamgx.finalize()
    return


def bench_AMGX_CG(A_csr_scipy, x_np, y, timer):

    # in the beginning x is going to be all zeros.
    # remember to zero it out between runs

    pyamgx.initialize()
    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict({
    "config_version": 2,
            "determinism_flag": 1,
            "exception_handling" : 1,
            # "print_coloring_info": 1,
            "solver": {
                "monitor_residual": 1,
                # "print_solve_stats": 1,
                "convergence_analysis": 1,
                "solver": "PCG",
                # "solver_verbose":1,
                "relaxation_factor": 1,
                # "obtain_timings": 1,
                "convergence": "RELATIVE_INI",
                "tolerance": 1e-12,
                "norm": "L2",
                "max_iters": 500,
                "preconditioner": {
                    "solver": "AMG",
                    "algorithm": "CLASSICAL",
                    "smoother": {
                        "solver": "MULTICOLOR_GS",
                        "symmetric_GS": 1, 
                        "max_iters": 1,
                    },
                    "relaxation_factor": 1,
                    "presweeps": 1,
                    "postsweeps": 1,
                    "max_levels": 3,
                    "max_iters": 1
                    # "scope": "amg"
                    
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

    A.upload_CSR(A_csr_scipy)
    b.upload(y)
    x.upload(x_np)
    
    # Setup and solve system:
    for i in range (10):
        print("setup and solve system", flush=True)
        x.upload(np.zeros_like(x_np))
        timer.start_timer()
        solver.setup(A)
        solver.solve(b, x)
        timer.stop_timer("computeCG")
        # zero out x

    x.download(x_np)

    # print(f"Solution: {x_np[:10]}")

    # we compute a norm
    # Ax - b
    # norm = np.linalg.norm(cp.array(A_csr_scipy @ x_np) - cp.array(y))
    # print(f"Norm: {norm}")
    # Download solution

    # timer.update_additional_info(timer.get_additional_info() + f"L2 Norm: {norm}")


    # Clean up
    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    pyamgx.finalize()
    return


def run_AMGX_benchmark(nx: int, ny: int, nz: int, save_folder):

    meta_data , A_csr, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    x = np.zeros(meta_data.num_rows, dtype=np.float64)
    A_csr_scipy =  sparse.csr_matrix((A_csr.data.get(), A_csr.indices.get(), A_csr.indptr.get()), shape=A_csr.shape)

    nnz = meta_data.nnz

    
    version_name = f"AMGX"

    matrix_timer = gpu_timer(
        version_name = version_name,
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nnz,
        folder_path = save_folder
    )

    vector_timer = gpu_timer(
        version_name = version_name,
        ault_node = ault_node,
        matrix_type = "3d_27pt",
        nx = nx,
        ny = ny,
        nz = nz,
        nnz = nx*ny*nz,
        folder_path = save_folder
    )


    print(f"Running AMGX benchmark for {nx}x{ny}x{nz}", flush=True)
    # bench_AMGX_SymGS(nx, ny, nz, A_csr_scipy, x, y, matrix_timer)
    bench_AMGX_CG(A_csr_scipy, x, y, matrix_timer)

    matrix_timer.destroy_timer()
    vector_timer.destroy_timer()

    return


def get_num_its(nx, ny, nz):
    # create Matrix
    meta_data , A_csr, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    x = np.zeros(meta_data.num_rows, dtype=np.float64)

    
    num_iterations = 1
    rr_norm = 10

    threshold_rr_norm = getSymGS_rrNorm_zero_based(nx, ny, nz)

    while rr_norm > threshold_rr_norm:
        print(f"num_iterations: {num_iterations}")
        run_AMGX_SymGS(A_csr, x, y, num_iterations)
        rr_norm = np.linalg.norm(cp.array(A_csr @ cp.array(x)) - cp.array(y)) / np.linalg.norm(cp.array(y))
        # print(f"rr_norm: {rr_norm}")
        num_iterations += 1
    
    print(f"size: {nx}x{ny}x{nz} converged at {num_iterations-1} iterations")

def check_num_its(nx, ny, nz):
    # create Matrix
    meta_data , A_csr, y = generations.generate_cupy_csr_problem(nx, ny, nz)
    x = np.zeros(meta_data.num_rows, dtype=np.float64)

    num_iterations = num_its_zerobased_AMGX[(nx,ny,nz)]
    rr_norm = 10
    error_counter = 0

    threshold_rr_norm = getSymGS_rrNorm_zero_based(nx, ny, nz)

    for i in range(20):

        run_AMGX_SymGS(A_csr, x, y, num_iterations)
        rr_norm = np.linalg.norm(cp.array(A_csr @ cp.array(x)) - cp.array(y)) / np.linalg.norm(cp.array(y))
        # print(f"rr_norm: {rr_norm}")

        if rr_norm > threshold_rr_norm:
            error_counter += 1
    
    print(f"size: {nx}x{ny}x{nz} failed to converge at {num_iterations} iterations {error_counter} times")
    

if __name__ == "__main__":
    # get_num_its(2,2,2)
    # get_num_its(4,4,4)
    # get_num_its(8,8,8)
    # get_num_its(16,16,16)
    # get_num_its(32,32,32)
    # get_num_its(64,64,64)
    # get_num_its(128,64,64)
    # the following kill the execution
    # get_num_its(128,128,64)
    # get_num_its(128,128,128)

    # check_num_its(2,2,2)
    # check_num_its(4,4,4)
    # check_num_its(8,8,8)
    # check_num_its(16,16,16)
    # check_num_its(32,32,32)
    # check_num_its(64,64,64)
    # check_num_its(128,64,64)

    # I just want to run the SymGS to see the colors
    # first generate the np matrix

    # A_np = generate_stencil_matrix(2, Shape.SQUARE, 1, [4, 4])
    # print(A_np)

    # A_csr = sparse.csr_matrix(A_np)
    # x_np = np.zeros(A_np.shape[0], dtype=np.float64)
    # y = np.zeros(A_np.shape[0], dtype=np.float64)

    # # fill up y with 8.0 - nnz in row
    # for i in range(A_np.shape[0]):
    #     y[i] = 8.0 - A_csr[i].nnz
    
    # run_AMGX_SymGS(A_csr, x_np, y, 1)

    meta_data , A_csr, y = generations.generate_cupy_csr_problem(64, 64, 64)
    x = np.zeros(meta_data.num_rows, dtype=np.float64)
    A_csr_scipy =  sparse.csr_matrix((A_csr.data.get(), A_csr.indices.get(), A_csr.indptr.get()), shape=A_csr.shape)


    run_AMGX_CG(A_csr_scipy, x, y)

