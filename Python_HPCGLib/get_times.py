from gpu_timing import gpu_timer
import torch
import time
import random

import testing
import HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.generations as generations
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.matlab_reference as matlab_reference
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BaseTorch as BaseTorch
import HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions.BasicStencil as BasicStencil

num_iterations = 5
do_tests = False
debug = True

sizes =[
    (8, 8, 8),
    (16, 16, 16),
    (32, 32, 32),
    # (64, 64, 64),
    # (128, 128, 128),
]

versions = [
    "BaseTorch",
    # "MatlabReference",
    # "BasicStencil",
]

methods = [
    # "computeSymGS",
    "computeSPMV",
    # "computeRestriction",
    # "computeMG",
    # "computeProlongation",
    # "computeCG",
    # "computeWAXPBY",
    # "computeDot",
]

matrix_types = [
    "3d_27pt"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#################################################################################################################
# run the tests for all of the versions
#################################################################################################################
if do_tests:
    print("Starting tests", flush=True)
    testing.run_tests(sizes, matrix_types, methods, versions)
    print("Tests finished", flush=True)

#################################################################################################################
# for each version we need the following
# because each version has it's own name and needs to be called!
#################################################################################################################
print("Starting timing", flush=True)
overall_start = time.time()

torch.cuda.synchronize()
for v in versions:
    if "BaseTorch" == v:
        for size in sizes:
            for matrix_type in matrix_types:

                # initialize the matrix and vectors (not included in measurments)
                A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
                x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
                
                a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
                alpha, beta = random.random(), random.random()

                nnz = A._nnz()

                matrix_timer = gpu_timer(
                    version_name = v,
                    ault_node = "41-44",
                    matrix_type = matrix_type,
                    nx = size[0],
                    ny = size[1],
                    nz = size[2],
                    nnz = nnz
                )

                vector_timer = gpu_timer(
                    version_name = v,
                    ault_node = "41-44",
                    matrix_type = matrix_type,
                    nx = size[0],
                    ny = size[1],
                    nz = size[2],
                    nnz = size[0]*size[1]*size[2]
                )

                # note: CG and MG include generations of matrices!
                # Also note: in the original HPCG the generations are all done in the main function, non of them in the MG routine.
                
                for m in methods:
                    if "computeCG" == m:
                        for i in range(num_iterations):

                            matrix_timer.start_timer()
                            BaseTorch.computeCG(size[0], size[1], size[2], A, y, x)
                            matrix_timer.stop_timer("computeCG")
                    
                    elif "computeMG" == m:
                        for i in range(num_iterations):

                            matrix_timer.start_timer()
                            BaseTorch.computeMG(size[0], size[1], size[2], A, y, x, 0)
                            matrix_timer.stop_timer("computeMG")
                    
                    elif "computeSymGS" == m:
                        for i in range(num_iterations):

                            matrix_timer.start_timer()
                            BaseTorch.computeSymGS(size[0], size[1], size[2], A, y, x)
                            matrix_timer.stop_timer("computeSymGS")
                    
                    elif "computeSPMV" == m:
                            for i in range(num_iterations):
                
                                matrix_timer.start_timer()
                                # careful! This way we only get "nice" numbers for the vectors, which does not reflect how spmv is used in the routines
                                # We might possibly want to read a bunch of y options from a file and use them here
                                BaseTorch.computeSPMV(size[0], size[1], size[2], A, y, x)
                                matrix_timer.stop_timer("computeSPMV")

                    elif "computeWAXPBY" == m:
                        for i in range(num_iterations):

                            vector_timer.start_timer()
                            BaseTorch.computeWAXPBY(alpha, a ,beta, b, x)
                            vector_timer.stop_timer("computeWAXPBY")
                    
                    elif "computeDot" == m:
                        for i in range(num_iterations):

                            vector_timer.start_timer()
                            BaseTorch.computeDot(a, b)
                            vector_timer.stop_timer("computeDot")

                    else:
                        print(f"WARNING: BaseTorch does not have a Mehtod {m} implemented for timing")

                matrix_timer.destroy_timer()
                vector_timer.destroy_timer()

    elif v == "MatlabReference":
        for size in sizes:
            for matrix_type in matrix_types:

                # initialize the matrix and vectors (not included in measurments)
                A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
                x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)

                a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
                alpha, beta = random.random(), random.random()

                nnz = A._nnz()

                matrix_timer = gpu_timer(
                    version_name = v,
                    ault_node = "41-44",
                    matrix_type = matrix_type,
                    nx = size[0],
                    ny = size[1],
                    nz = size[2],
                    nnz = nnz
                )

                vector_timer = gpu_timer(
                    version_name = v,
                    ault_node = "41-44",
                    matrix_type = matrix_type,
                    nx = size[0],
                    ny = size[1],
                    nz = size[2],
                    nnz = size[0]*size[1]*size[2]
                )

                # note: CG and MG include generations of matrices!
                # Also note: in the original HPCG the generations are all done in the main function, non of them in the MG routine.
                
                for m in methods:

                    if "computeCG" == m:
                        for i in range(num_iterations):

                            matrix_timer.start_timer()
                            matlab_reference.computeCG(A, y, x)
                            matrix_timer.stop_timer("computeCG")
                
                    
                    elif "computeSymGS" == m:
                        for i in range(num_iterations):

                            matrix_timer.start_timer()
                            matlab_reference.computeSymGS(A, y)
                            matrix_timer.stop_timer("computeSymGS")
                    
                    elif "computeSPMV" == m:
                            for i in range(num_iterations):

                                y = y.unsqueeze(1) if y.dim() == 1 else y
                                matrix_timer.start_timer()
                                # careful! This way we only get "nice" numbers for the vectors, which does not reflect how spmv is used in the routines
                                # We might possibly want to read a bunch of y options from a file and use them here
                                matlab_reference.computeSPMV(A, y)
                                matrix_timer.stop_timer("computeSPMV")
                                y = y.squeeze(1) if y.dim() > 1 else y

                    elif "computeWAXPBY"  == m:
                        for i in range(num_iterations):

                            vector_timer.start_timer()
                            matlab_reference.computeWAXPBY(a, x, beta, y)
                            vector_timer.stop_timer("computeWAXPBY")
                    
                    elif "computeDot" == m:
                        for i in range(num_iterations):

                            vector_timer.start_timer()
                            matlab_reference.computeDot(a, b)
                            vector_timer.stop_timer("computeDot")
                    
                    else:
                        print(f"WARNING: MatlabReference does not have a Mehtod {m} implemented for timing")

                matrix_timer.destroy_timer()
                vector_timer.destroy_timer()
    
    elif v == "BasicStencil":
        for size in sizes:
            for matrix_type in matrix_types:

                # initialize the matrix and vectors (not included in measurments)
                A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
                banded_A = BasicStencil.convert_A_to_Band_matrix(size[0], size[1], size[2], A)
                x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)

                a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
                alpha, beta = random.random(), random.random()

                nnz = A._nnz()

                matrix_timer = gpu_timer(
                    version_name = v,
                    ault_node = "41-44",
                    matrix_type = matrix_type,
                    nx = size[0],
                    ny = size[1],
                    nz = size[2],
                    nnz = nnz
                )

                vector_timer = gpu_timer(
                    version_name = v,
                    ault_node = "41-44",
                    matrix_type = matrix_type,
                    nx = size[0],
                    ny = size[1],
                    nz = size[2],
                    nnz = size[0]*size[1]*size[2]
                )

                for m in methods:

                    if m == "computeSPMV":
                        for i in range(num_iterations):
                            matrix_timer.start_timer()
                            BasicStencil.computeSPMV(size[0], size[1], size[2], banded_A, y, x)
                            matrix_timer.stop_timer(m)
                    else:
                        print(f"WARNING: {v} does not have a method {m} implemented for timing")
                
                matrix_timer.destroy_timer()
                vector_timer.destroy_timer()

    else:
        print(f"WARNING: {v} not implemented for timing")

torch.cuda.synchronize()
overall_end = time.time()

time_elapsed = overall_end - overall_start
minutes, seconds = divmod(time_elapsed, 60)

print("Timing finished")
print(f"Timing took: {time_elapsed}")
#################################################################################################################