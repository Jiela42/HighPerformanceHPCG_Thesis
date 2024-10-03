from gpu_timing import gpu_timer
import generations
import torch
import time
import testing
import matlab_reference
import BaseTorch

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
    "MatlabReference",
]

methods = [
    "computeSymGS",
    "computeSPMV",
    # "computeRestriction",
    "computeMG",
    # "computeProlongation",
    # "computeCG",
    # "computeWAXPBY",
    "computeDot",
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
    testing.test(sizes, matrix_types, methods, versions)

#################################################################################################################
# for each version we need the following
# because each version has it's own name and needs to be called!
#################################################################################################################
print("Starting timing", flush=True)
overall_start = time.time()

torch.cuda.synchronize()

if "BaseTorch" in versions:
    for size in sizes:
        for matrix_type in matrix_types:

            # initialize the matrix and vectors (not included in measurments)
            A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
            x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)

            a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)

            nnz = A._nnz()

            matrix_timer = gpu_timer(
                version_name = "BaseTorch",
                ault_node = "41-44",
                matrix_type = matrix_type,
                nx = size[0],
                ny = size[1],
                nz = size[2],
                nnz = nnz
            )

            vector_timer = gpu_timer(
                version_name = "BaseTorch",
                ault_node = "41-44",
                matrix_type = matrix_type,
                nx = size[0],
                ny = size[1],
                nz = size[2],
                nnz = size[0]*size[1]*size[2]
            )

            # note: CG and MG include generations of matrices!
            # Also note: in the original HPCG the generations are all done in the main function, non of them in the MG routine.
            

            if "computeCG" in methods:
                for i in range(num_iterations):

                    matrix_timer.start_timer()
                    BaseTorch.computeCG(size[0], size[1], size[2])
                    matrix_timer.stop_timer("computeCG")
            
            if "computeMG" in methods:
                for i in range(num_iterations):

                    matrix_timer.start_timer()
                    BaseTorch.computeMG(size[0], size[1], size[2], A, y, x, 0)
                    matrix_timer.stop_timer("computeMG")
            
            if "computeSymGS" in methods:
                for i in range(num_iterations):

                    matrix_timer.start_timer()
                    BaseTorch.computeSymGS(size[0], size[1], size[2], A, y, x)
                    matrix_timer.stop_timer("computeSymGS")
            
            if "computeSPMV" in methods:
                    for i in range(num_iterations):
        
                        matrix_timer.start_timer()
                        # careful! This way we only get "nice" numbers for the vectors, which does not reflect how spmv is used in the routines
                        # We might possibly want to read a bunch of y options from a file and use them here
                        BaseTorch.computeSPMV(size[0], size[1], size[2], A, y, x)
                        matrix_timer.stop_timer("computeSPMV")

            if "computeWAXPBY" in methods:
                for i in range(num_iterations):

                    vector_timer.start_timer()
                    # BaseTorch.computeWAXPBY(a, x ,b, y, w)
                    vector_timer.stop_timer("computeWAXPBY")
            
            if "computeDot" in methods:
                for i in range(num_iterations):

                    vector_timer.start_timer()
                    BaseTorch.computeDot(a, b)
                    vector_timer.stop_timer("computeDot")

            matrix_timer.destroy_timer()
            vector_timer.destroy_timer()

if "MatlabReference" in versions:
    for size in sizes:
        for matrix_type in matrix_types:

            # initialize the matrix and vectors (not included in measurments)
            A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
            x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)

            a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
        
            nnz = A._nnz()

            matrix_timer = gpu_timer(
                version_name = "MatlabReference",
                ault_node = "41-44",
                matrix_type = matrix_type,
                nx = size[0],
                ny = size[1],
                nz = size[2],
                nnz = nnz
            )

            vector_timer = gpu_timer(
                version_name = "MatlabReference",
                ault_node = "41-44",
                matrix_type = matrix_type,
                nx = size[0],
                ny = size[1],
                nz = size[2],
                nnz = size[0]*size[1]*size[2]
            )

            # note: CG and MG include generations of matrices!
            # Also note: in the original HPCG the generations are all done in the main function, non of them in the MG routine.
            

            if "computeCG" in methods:
                for i in range(num_iterations):

                    matrix_timer.start_timer()
                    matlab_reference.computeCG(size[0], size[1], size[2])
                    matrix_timer.stop_timer("computeCG")
            
            # if "computeMG" in methods:
                # for i in range(num_iterations):

                #     matrix_timer.start_timer()
                #     matlab_reference.computeMG(size[0], size[1], size[2], A, y, x, 0)
                #     matrix_timer.stop_timer("computeMG")
            
            if "computeSymGS" in methods:
                for i in range(num_iterations):

                    matrix_timer.start_timer()
                    matlab_reference.computeSymGS(A, y)
                    matrix_timer.stop_timer("computeSymGS")
            
            if "computeSPMV" in methods:
                    for i in range(num_iterations):
        
                        matrix_timer.start_timer()
                        # careful! This way we only get "nice" numbers for the vectors, which does not reflect how spmv is used in the routines
                        # We might possibly want to read a bunch of y options from a file and use them here
                        matlab_reference.computeSPMV(A, y)
                        matrix_timer.stop_timer("computeSPMV")

            if "computeWAXPBY" in methods:
                for i in range(num_iterations):

                    vector_timer.start_timer()
                    # matlab_reference.computeWAXPBY(a, x ,b, y, w)
                    vector_timer.stop_timer("computeWAXPBY")
            
            if "computeDot" in methods:
                for i in range(num_iterations):

                    vector_timer.start_timer()
                    matlab_reference.computeDot(a, b)
                    vector_timer.stop_timer("computeDot")

            matrix_timer.destroy_timer()
            vector_timer.destroy_timer()

torch.cuda.synchronize()
overall_end = time.time()

time_elapsed = overall_end - overall_start
minutes, seconds = divmod(time_elapsed, 60)

print("Timing finished")
print(f"Timing took: {time_elapsed}")
#################################################################################################################