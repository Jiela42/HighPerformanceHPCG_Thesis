# this file is different, because it benchmarks AMGX, which is not part of the HPCG_versions
# the reason for this is that AMGX takes different inputs
# and the way it's setup we don't want to mess with the scope of initialization


import HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.BenchmarkingLib.gpu_timing import gpu_timer
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import ault_node, num_bench_iterations

import pyamgx
import scipy.sparse as sparse
import numpy as np
import cupy as cp

def bench_AMGX_SymGS(nx, ny, nz, A_csr_scipy, x_np, y, timer):

    # in the beginning x is going to be all zeros.
    # remember to zero it out between runs

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
                "symmetric_GS": 1, 
                "max_iters": 2,
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
    for i in range (num_bench_iterations):
        x.upload(np.zeros_like(x_np))
        timer.start_timer()
        solver.setup(A)
        solver.solve(b, x)
        timer.stop_timer("computeSymGS")
        # zero out x

    # Download solution
    x.download(x_np)

    # print(f"Solution: {x_np[:10]}")

    # we compute a norm
    # Ax - b
    norm = np.linalg.norm(cp.array(A_csr_scipy @ x_np) - cp.array(y))
    # print(f"Norm: {norm}")

    timer.update_additional_info(timer.get_additional_info() + f"L2 Norm: {norm}")


    # Clean up:
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


    nnz = meta_data.nnz

    version_name = "AMGX 2 iterations"

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

    x = np.zeros(meta_data.num_rows, dtype=np.float64)
    A_csr_scipy =  sparse.csr_matrix((A_csr.data.get(), A_csr.indices.get(), A_csr.indptr.get()), shape=A_csr.shape)

    bench_AMGX_SymGS(nx, ny, nz, A_csr_scipy, x, y, matrix_timer)

    matrix_timer.destroy_timer()
    vector_timer.destroy_timer()

    return
