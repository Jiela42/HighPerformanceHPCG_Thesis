import torch
from typing import List, Tuple
import generations
import matlab_reference
import BaseTorch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
error_tolerance = 1e-9
debug = True


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

    for i in range(tested.size(0)):
        print(f"i: {i} should be: {control[i].item()} but was: {tested[i]}")

    # print the wrong elements next to each other
    print(f"control size: {control.size()}")
    print(f"tested size: {tested.size()}")
    for i in range(tested.size(0)):
        if not torch.isclose(tested[i], control[i],atol=error_tolerance):
            print(f"i: {i} should be: {control[i]} but was: {tested[i]}")

def test(sizes: List[Tuple[int, int, int]], matrix_types: List[str], methods: List[str], Versions: List[str]) -> None:

    if "BaseTorch" in Versions:
        for size in sizes:

            A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
            # print(y)
            y_original = y.clone()
            x = torch.zeros(size[0]*size[1]*size[2], device=device, dtype=torch.float64)
            a,b = torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64), torch.rand(size[0]*size[1]*size[2], device=device, dtype=torch.float64)

            if "computeCG" in methods:
                tested_x = BaseTorch.computeCG(A, b, x)
                control_x = matlab_reference.computeCG(A, b, x)
                if not torch.allclose(tested_x, control_x, atol=error_tolerance):
                    print_differeing_vectors(tested_x, control_x)
                    raise AssertionError(f"Error in computeCG for size {size}, version BaseTorch")
            
            if "computeMG" in methods:
                BaseTorch.computeMG(size[0], size[1], size[2], A, y, x, 0)
                tested_y = x
                control_y = matlab_reference.computeMG(A, y)

                if not torch.allclose(tested_y, control_y, atol=error_tolerance):
                    print_differeing_vectors(tested_y, control_y)
                    raise AssertionError(f"Error in computeMG for size {size}, version BaseTorch")

            if "computeDot" in methods:
                tested_dot = BaseTorch.computeDot(a, b)
                control_dot = matlab_reference.computeDot(a, b)

                if not torch.allclose(tested_dot, control_dot, atol=error_tolerance):
                    print_differeing_vectors(tested_dot, control_dot)
                    raise AssertionError(f"Error in computeDot for size {size}, version BaseTorch")
            
            if "computeSymGS" in methods:

                # control_x_greg = matlab_reference.greg_symGS(A, y)
                # control_x_greg = control_x_greg.squeeze() if control_x_greg.dim() > 1 else control_x_greg

                control_x = matlab_reference.computeSymGS(A, y)
                BaseTorch.computeSymGS(size[0], size[1], size[2], A, y, x)
                tested_x = x

                # y_test_matlab = torch.sparse.mm(A, control_x.unsqueeze(1)).squeeze()
                # y_test = torch.sparse.mm(A, tested_x.unsqueeze(1)).squeeze()
                # y_test_greg = torch.sparse.mm(A, control_x_greg.unsqueeze(1)).squeeze()

                # if not torch.allclose(y_test_matlab, y, atol=error_tolerance):
                #     print("SymGS: y_test_matlab also failes the test")
                    # print_differeing_vectors(y_test_matlab, y_original)

                # if not torch.allclose(control_x, control_x_greg, atol=error_tolerance):
                #     print_differeing_vectors(control_x, control_x_greg)
                #     print(f"SymGS: matlab and greg do not agree")
                # else:
                #     print("SymGS: matlab and greg agree")

                if not torch.allclose(y, y_original, atol=error_tolerance):
                    raise AssertionError(f"ComputeSymGS changed y")


                # print(f"y_test size: {y_test.size()}")
                # print(f"y size: {y.size()}")
                # print(f"y_test_matlab size: {y_test_matlab.size()}")
                # print(f"y_test_greg size: {y_test_greg.size()}")
                # print(f"y_original size: {y_original.size()}")
                # print(f"tested_x size: {tested_x.size()}")
                # print(f"control_x size: {control_x.size()}")
                # print(f"control_x_greg size: {control_x_greg.size()}")


                # distance_BaseTorch = torch.norm(y_test-y)
                # distance_Matlab = torch.norm(y_test_matlab-y)
                # distance_greg = torch.norm(y_test_greg-y)

                # print(f"distance_BaseTorch: {distance_BaseTorch}")
                # print(f"distance_Matlab: {distance_Matlab}")
                # print(f"distance_greg: {distance_greg}")

                # # check the percision of the matrices and results and print them
                # print(f"Percision of A: {A.dtype}")
                # print(f"Percision of y: {y.dtype}")
                # print(f"Percision of x: {x.dtype}")
                # print(f"Percision of control_x: {control_x.dtype}")
                # print(f"Percision of control_x_greg: {control_x_greg.dtype}")
                # print(f"Percision of y_test: {y_test.dtype}")
                # print(f"Percision of y_test_matlab: {y_test_matlab.dtype}")
                # print(f"Percision of y_test_greg: {y_test_greg.dtype}")
                
                # print(y)
                # print(y_test_greg)

                # if not torch.allclose(y, y_test_greg, atol=error_tolerance):
                #     print_differeing_vectors(y_test_greg, y)
                #     raise AssertionError(f"Error in computeSymGS for size {size}, version BaseTorch")

                # if distance_BaseTorch > distance_Matlab:
                #     raise AssertionError(f"Error in computeSymGS for size {size}, version BaseTorch, distance_BaseTorch: {distance_BaseTorch}, distance_Matlab: {distance_Matlab}")

                # if not torch.allclose(y_test, y, atol=error_tolerance):
                #     # print_differeing_vectors(y_test, y)
                #     raise AssertionError(f"Error in computeSymGS for size {size}, version BaseTorch")

                if not torch.allclose(tested_x, control_x, atol=error_tolerance):
                    print_differeing_vectors(tested_x, control_x)
                    raise AssertionError(f"(x comparison) Error in computeSymGS for size {size}, version BaseTorch")
                elif debug:
                    print("SymGS: BaseTorch and Matlab agree")

                # if not torch.allclose(tested_x, control_x_greg, atol=error_tolerance):
                #     print_differeing_vectors(tested_x, control_x_greg)
                #     raise AssertionError(f"(x comparison) Error in computeSymGS for size {size}, version BaseTorch")
                # else:
                #     print("BaseTorch and greg agree")
                    

            if "computeSPMV" in methods:
                BaseTorch.computeSPMV(size[0], size[1], size[2], A, y, x)
                tested_solution = x
                control_solution = matlab_reference.computeSPMV(A, y)

                if not torch.allclose(y, y_original, atol=error_tolerance):
                    raise AssertionError(f"ComputeSPMV changed y")
                
                # differences = (tested_solution - control_solution).abs()
                # max_diff = differences.max()
                # print(differences)

                tested_solution = tested_solution.squeeze() if tested_solution.dim() > 1 else tested_solution
                control_solution = control_solution.squeeze() if control_solution.dim() > 1 else control_solution

                if not torch.allclose(tested_solution, control_solution, atol=error_tolerance):
                    print("MMMMMIIIIAUUUUU")
                    print(f"tested solution dims: {tested_solution.size()}")
                    print(f"control solution dims: {control_solution.size()}")
                    # print("Difference:", max_diff)
                    # print("Index of max difference:", torch.argmax(differences))
                    # print_differeing_vectors(tested_solution, control_solution)
                    raise AssertionError(f"Error in computeSPMV for size {size}, version BaseTorch")
                elif debug:
                    print("SPMV: BaseTorch and Matlab agree")

    print("----ALL TESTS PASSED----")


def symGS_mini_test():
    
    A = gen_stencil(16)
    x = torch.ones(16, device=device, dtype=torch.float64)
    y = A @ x
    empty_x = torch.zeros(16, device=device, dtype=torch.float64)
    # print(y)

    # make A sparse
    A_sparse_coo = A.to_sparse().to(dtype=torch.float64, device=device)

    x_greg = matlab_reference.greg_symGS(A_sparse_coo, y)
    x_matlab = matlab_reference.computeSymGS(A_sparse_coo, y)
    BaseTorch.computeSymGS(2, 2, 4, A_sparse_coo, y, empty_x)
    x_BaseTorch = empty_x

    # print(x_greg)
    # print(x_matlab)
    # print(x_BaseTorch)

    # for i in range(16):
    #     print(f"greg: {x_greg[i].item()}, matlab: {x_matlab[i].item()}, BaseTorch: {x_BaseTorch[i].item()}")

    x_greg = x_greg.squeeze() if x_greg.dim() > 1 else x_greg

    if torch.allclose(x_greg, x_matlab, atol=error_tolerance):
        print("SymGS_Mini_testgreg and matlab are the same")
    if torch.allclose(x_greg, x_BaseTorch, atol=error_tolerance):
        print("SymGS_Mini_test: greg and BaseTorch are the same")
    # else:
        # print("appearently non of them are the same")
        # print(x_greg.dtype)
        # print(x_BaseTorch.dtype)
        # print(x_matlab.dtype)
        # print(x_greg.size())
        # print(x_BaseTorch.size())
        # print(x_matlab.size())

# sizes =[
#     (8, 8, 8),
#     (16, 16, 16),
#     # (32, 32, 32),
#     # (64, 64, 64),
#     # (128, 128, 128),
# ]

# versions = [
#     "BaseTorch",
#     # "MatlabReference",
# ]

# methods = [
#     "computeSymGS",
#     "computeSPMV",
#     # "computeRestriction",
#     # "computeMG",
#     # "computeProlongation",
#     # "computeCG",
#     # "computeWAXPBY",
#     "computeDot",
# ]

# matrix_types = [
#     "3d_27pt"
# ]

# symGS_mini_test()
# test(sizes, matrix_types, methods, versions)

# def run_tests(sizes, matrix_types, methods, versions):
#     symGS_mini_test()
#     test(sizes, matrix_types, methods, versions)


# print(A)