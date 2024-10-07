import torch
from typing import List, Tuple

import torch.cuda
import generations
import matlab_reference
import BaseTorch
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
error_tolerance = 1e-9
debug = False


def gen_stencil(n):
    A = torch.zeros([n,n], device=device, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
                if (np.abs(i-j) % 4) < 2:
                    A[i,j] = 1
    return A

def print_differeing_vectors(tested: torch.Tensor, control: torch.Tensor) -> None:

    tested = tested.squeeze() if tested.dim() > 1 else tested
    control = control.squeeze() if control.dim() > 1 else control

    if tested.dtype != control.dtype:
        raise AssertionError(f"Dtype of the vectors is different: {tested.dtype} vs {control.dtype}")

    if tested.size(0) != control.size(0):
        raise AssertionError(f"Size of the vectors is different: {tested.size(0)} vs {control.size(0)}")

    # for i in range(tested.size(0)):
    #     print(f"i: {i} should be: {control[i].item()} but was: {tested[i]}")

    # print the wrong elements next to each other
    print(f"control size: {control.size()}")
    print(f"tested size: {tested.size()}")
    for i in range(tested.size(0)):
        if not torch.isclose(tested[i], control[i],atol=error_tolerance):
            print(f"i: {i} should be: {control[i]} but was: {tested[i]}")

def test(sizes: List[Tuple[int, int, int]], matrix_types: List[str], methods: List[str], Versions: List[str]) -> None:

    for size in sizes:

        A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
        # print(y)
        y_original = y.clone()
        A_original = A.clone()
        x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
        a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
        
        if "BaseTorch" in Versions:

            if "computeCG" in methods:
                BaseTorch.computeCG_no_preconditioning(size[0], size[1], size[2], A, b, x)
                tested_x = x.clone()
                matlab_reference.computeCG(A, b, x)
                control_x = x.clone()
                
                if not torch.allclose(y, y_original, atol=error_tolerance):
                    raise AssertionError(f"BaseTorch ComputeCG changed y")
                
                if debug:
                    if not torch.allclose(A.to_dense(), A_original.to_dense(), atol=error_tolerance):
                        raise AssertionError(f"BaseTorch ComputeCG changed A")

                if not torch.allclose(tested_x, control_x, atol=error_tolerance):
                    print_differeing_vectors(tested_x, control_x)
                    raise AssertionError(f"Error in BaseTorch computeCG for size {size}, version BaseTorch")

            
            # if "computeMG" in methods:
            #     BaseTorch.computeMG(size[0], size[1], size[2], A, y, x, 0)
            #     tested_y = x.clone()
            #     control_y = matlab_reference.computeMG(A, y)

            #     if not torch.allclose(tested_y, control_y, atol=error_tolerance):
            #         print_differeing_vectors(tested_y, control_y)
            #         raise AssertionError(f"Error in BaseTorch computeMG for size {size}, version BaseTorch")

            if "computeDot" in methods:
                tested_dot = BaseTorch.computeDot(a, b)
                control_dot = matlab_reference.computeDot(a, b)

                if not torch.allclose(tested_dot, control_dot, atol=error_tolerance):
                    print_differeing_vectors(tested_dot, control_dot)
                    raise AssertionError(f"Error in BaseTorch computeDot for size {size}, version BaseTorch")
            
            if "computeSymGS" in methods:

                control_x = matlab_reference.computeSymGS(A, y)
                BaseTorch.computeSymGS(size[0], size[1], size[2], A, y, x)
                tested_x = x.clone()

                if not torch.allclose(y, y_original, atol=error_tolerance):
                    raise AssertionError(f"BaseTorch ComputeSymGS changed y")
                
                if debug:
                    if not torch.allclose(A.to_dense(), A_original.to_dense(), atol=error_tolerance):
                        raise AssertionError(f"BaseTorch ComputeSymGS changed A")

                if not torch.allclose(tested_x, control_x, atol=error_tolerance):
                    print_differeing_vectors(tested_x, control_x)
                    raise AssertionError(f"(x comparison) Error in BaseTorch computeSymGS for size {size}, version BaseTorch")
                elif debug:
                    print("BaseTorch SymGS: BaseTorch and Matlab agree")
                    

            if "computeSPMV" in methods:
                BaseTorch.computeSPMV(size[0], size[1], size[2], A, y, x)
                tested_solution = x.clone()

                y = y.unsqueeze(1) if y.dim() == 1 else y                
                control_solution = matlab_reference.computeSPMV(A, y)

                y = y.squeeze() if y.dim() > 1 else y
                if not torch.allclose(y, y_original, atol=error_tolerance):
                    raise AssertionError(f"ComputeSPMV changed y")

                control_solution = control_solution.squeeze() if control_solution.dim() > 1 else control_solution
                tested_solution = tested_solution.squeeze() if tested_solution.dim() > 1 else tested_solution

                if not torch.allclose(tested_solution, control_solution, atol=error_tolerance):
                    raise AssertionError(f"Error in BaseTorch computeSPMV for size {size}, version BaseTorch")
                elif debug:
                    print("BaseTorch SPMV: BaseTorch and Matlab agree", flush=True)
            
            if "computeWAXPBY" in methods:
                alpha, beta = random.random(), random.random()
                BaseTorch.computeWAXPBY(alpha, a, beta, b, x)
                tested_solution = x.clone()

                control_solution = matlab_reference.computeWAXPBY(alpha, a, beta, b)

                if not torch.allclose(tested_solution, control_solution, atol=error_tolerance):
                    print_differeing_vectors(tested_solution, control_solution)
                    raise AssertionError(f"Error in BaseTorch computeWAXPBY for size {size}, version BaseTorch")
                elif debug:
                    print("BaseTorch WAXPBY: BaseTorch and Matlab agree", flush=True)

    print("----ALL TESTS PASSED----")


def symGS_mini_test(versions):
    
    A = gen_stencil(16)
    x = torch.ones(16, device=device, dtype=torch.float64)
    y = A @ x
    empty_x = torch.zeros(16, device=device, dtype=torch.float64)

    # make A sparse
    A_sparse_coo = A.to_sparse().to(dtype=torch.float64, device=device)

    x_greg = matlab_reference.greg_symGS(A_sparse_coo, y)
    x_greg = x_greg.squeeze() if x_greg.dim() > 1 else x_greg

    x_matlab = matlab_reference.computeSymGS(A_sparse_coo, y)

    if not torch.allclose(x_greg, x_matlab, atol=error_tolerance):
        raise AssertionError(f"SymGS_Mini_test: greg and matlab are different")
    elif debug:
        print("SymGS_Mini_testgreg and matlab are the same", flush=True)
    
    if "BaseTorch" in versions:
        BaseTorch.computeSymGS(2, 2, 4, A_sparse_coo, y, empty_x)
        x_BaseTorch = empty_x.clone()

        if not torch.allclose(x_greg, x_BaseTorch, atol=error_tolerance):
            # print_differeing_vectors(x_greg, x_BaseTorch)
            print("x_greg shape: ", x_greg.shape)
            print("x_BaseTorch shape: ", x_BaseTorch.shape)
            raise AssertionError(f"SymGS_Mini_test: greg and BaseTorch are different")
        elif debug:
            print("SymGS_Mini_test: greg and BaseTorch are the same", flush=True)

# sizes =[
#     (8, 8, 8),
#     # (16, 16, 16),
#     # (32, 32, 32),
#     # (64, 64, 64),
#     # (128, 128, 128),
# ]

# versions = [
#     "BaseTorch",
#     # "MatlabReference",
# ]

# methods = [
#     # "computeSymGS",
#     # "computeSPMV",
#     # "computeRestriction",
#     # "computeMG",
#     # "computeProlongation",
#     # "computeCG",
#     "computeWAXPBY",
#     # "computeDot",
# ]

# matrix_types = [
#     "3d_27pt"
# ]

# if "computeSymGS" in methods:
#     symGS_mini_test(versions)
# test(sizes, matrix_types, methods, versions)

def run_tests(sizes, matrix_types, methods, versions):
    if "computeSymGS" in methods:
        symGS_mini_test(versions)
    test(sizes, matrix_types, methods, versions)


# print(A)