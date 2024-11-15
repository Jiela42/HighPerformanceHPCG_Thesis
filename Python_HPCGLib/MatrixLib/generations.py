import numpy as np
# import scipy.sparse as sp
# from scipy.sparse import lil_matrix
import torch
from typing import Tuple
import itertools

from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix

store = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_y_forHPCG_problem(nx:int, ny:int, nz:int)-> torch.Tensor:

    y = torch.zeros(nx*ny*nz, device=device, dtype=torch.float64)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                nnz_i = 0
                for sz in range(-1, 2):
                    if iz+sz > -1 and iz+sz < nz:
                        for sy in range(-1, 2):
                            if iy+sy > -1 and iy+sy < ny:
                                for sx in range(-1, 2):
                                    if ix+sx > -1 and ix+sx < nx:
                                        nnz_i += 1
                y[i] = 26.0 - nnz_i


    return y

def generate_torch_csr_problem(nx: int, ny: int, nz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    A = CSRMatrix()
    A.create_3d27pt_CSRMatrix(nx, ny, nz)
    crow_indices, col_indices, values = A.to_torch()

    y = generate_y_forHPCG_problem(nx, ny, nz)    

    return crow_indices, col_indices, values, y

def generate_torch_coo_problem(nx: int, ny: int, nz: int) -> Tuple[torch.sparse.Tensor, torch.Tensor]:

    row_indices = []
    col_indices = []
    values = []

    y = torch.zeros(nx * ny * nz, device=device, dtype=torch.float64)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx * iy + nx * ny * iz
                nnz_i = 0
                for sz in range(-1, 2):
                    if iz + sz > -1 and iz + sz < nz:
                        for sy in range(-1, 2):
                            if iy + sy > -1 and iy + sy < ny:
                                for sx in range(-1, 2):
                                    if ix + sx > -1 and ix + sx < nx:
                                        j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz)
                                        row_indices.append(i)
                                        col_indices.append(j)
                                        if i == j:
                                            values.append(26.0)
                                        else:
                                            values.append(-1.0)
                                        nnz_i += 1
                y[i] = 26.0 - (float(nnz_i -1))

    row_indices = torch.tensor(row_indices, device=device, dtype=torch.int64)
    col_indices = torch.tensor(col_indices, device=device, dtype=torch.int64)
    values = torch.tensor(values, device=device, dtype=torch.float64)

    A = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]), values, (nx * ny * nz, nx * ny * nz), device=device, dtype=torch.float64)
    A = A.coalesce()

    return A, y

"""
def generate_lil_problem(nx: int, ny: int, nz: int) -> Tuple[lil_matrix, np.ndarray]:

    A = sp.lil_matrix((nx*ny*nz, nx*ny*nz))
    y = np.zeros(nx*ny*nz)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                for sz in range(-1, 2):
                    if iz+sz > -1 and iz+sz < nz:
                        for sy in range(-1, 2):
                            if iy+sy > -1 and iy+sy < ny:
                                for sx in range(-1, 2):
                                    if ix+sx > -1 and ix+sx < nx:
                                        j = ix+sx + nx*(iy+sy) + nx*ny*(iz+sz)
                                        if i == j:
                                            A[i, j] = 26.0
                                        else:
                                            A[i, j] = -1.0
                y[i] = 26.0 - A[i].sum()
    
    return A, y
"""

def generate_coarse_problem(nxf: int, nyf: int, nzf: int) -> Tuple[np.ndarray, torch.tensor, torch.tensor]:

    if (nxf % 2 == 1 or nyf % 2 == 1 or nzf % 2 == 1):
        raise ValueError("nx, ny, and nz must be even")

    nxc = nxf // 2
    nyc = nyf // 2
    nzc = nzf // 2

    f2c_op = torch.zeros(nxc*nyc*nzc, dtype=torch.int64, device=device)

    coarse_num_rows = nxc * nyc * nzc

    for izc in range(nzc):
        izf = 2*izc
        for iyc in range(nyc):
            iyf = 2*iyc
            for ixc in range(nxc):
                ixf = 2*ixc
                current_coarse_row = ixc + nxc*iyc + nxc*nyc*izc

                current_fine_row = ixf + nxf*iyf + nxf*nyf*izf
                f2c_op[current_coarse_row] = current_fine_row

    # print(f2c_op)

    # print("We generate the new problem using the COO format")
    
    Ac, yc = generate_torch_coo_problem(nxc, nyc, nzc)

    return f2c_op, Ac, yc

def generate_coarse_csr_problem(nxf: int, nyf: int, nzf: int)-> Tuple[np.ndarray, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:

    if (nxf % 2 == 1 or nyf % 2 == 1 or nzf % 2 == 1):
        raise ValueError("nx, ny, and nz must be even")

    nxc = nxf // 2
    nyc = nyf // 2
    nzc = nzf // 2

    f2c_op = torch.zeros(nxc*nyc*nzc, dtype=torch.int64, device=device)

    coarse_num_rows = nxc * nyc * nzc

    for izc in range(nzc):
        izf = 2*izc
        for iyc in range(nyc):
            iyf = 2*iyc
            for ixc in range(nxc):
                ixf = 2*ixc
                current_coarse_row = ixc + nxc*iyc + nxc*nyc*izc

                current_fine_row = ixf + nxf*iyf + nxf*nyf*izf
                f2c_op[current_coarse_row] = current_fine_row

    # print(f2c_op)

    # print("We generate the new problem using the COO format")
    
    Ac_row_ptr, Ac_col_ptr, Ac_vals, yc = generate_torch_csr_problem(nxc, nyc, nzc)

    return f2c_op, Ac_row_ptr, Ac_col_ptr, Ac_vals, yc

def generate_Dense_Problem(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray]:

    # generate a np matrix with nx*ny*nz rows and columns
    A = np.zeros((nx*ny*nz, nx*ny*nz))
    y = np.zeros(nx*ny*nz)

    num_nnz_per_row = np.zeros(nx*ny*nz)
    col_idx_per_row = []
    stair_formation_rows = []
    num_zeros_before_diag = []
    num_zeros_before_z_jump_diag = []


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
                                            A[i, j] = 26.0
                                            # Ax = y
                                            # y[i] += 26.0 * x[j]
                                        else:
                                            A[i, j] = -1.0
                                            # y[i] -= 1.0 * x[j]
                                        num_nnz_per_row[i] += 1
                                        col_idx_i.append(j)
                col_idx_per_row.append(col_idx_i)
                y[i] = 26.0 - float(num_nnz_per_row[i])


    A_square = A @ A
    inverse_A = np.linalg.inv(A)
    neg_A = -A
    neg_A_A = neg_A @ A

    inverse_nnz = np.count_nonzero(inverse_A)

    # The following is for exprimenting with the stair formations and dependencies
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                if i-2 > 0:
                    if i - 1 > 0:
                        if A[i, i-1] == 0 and A[i, i-2] == 0:
                            stair_formation_rows.append(i)

                if A[i, i-1] == 0 and i > 1:
                    num_zeros = 1

                    while i - num_zeros -1 > 0 and (A[i, i-num_zeros-1] == 0):
                        num_zeros += 1
                    
                    num_zeros_before_diag.append(num_zeros)

                    if i % (nx * ny) == 0:
                        num_zeros_before_z_jump_diag.append(num_zeros)

    if store:
        A_square = np.vstack([np.arange(2, A_square.shape[0] + 2), A_square])
        # print the matrix to a file

        filename = "example_matrices/matrix_" + str(nx) + "x" + str(ny) + "x" + str(nz) + "_square.txt"
        np.savetxt(filename, A_square, fmt='%5g')

        inverse_A = np.vstack([np.arange(2, inverse_A.shape[0] + 2), inverse_A])
        filename = "example_matrices/matrix_" + str(nx) + "x" + str(ny) + "x" + str(nz) + "_inverse.txt"
        np.savetxt(filename, inverse_A, fmt='%5g')

        neg_A_A = np.vstack([np.arange(2, neg_A.shape[0] + 2), neg_A])
        filename = "example_matrices/matrix_" + str(nx) + "x" + str(ny) + "x" + str(nz) + "_negA_A.txt"
        np.savetxt(filename, neg_A_A, fmt='%5g')

        A = np.vstack([np.arange(2, A.shape[0] + 2), A])
        # print the matrix to a file
        if nz < 20 and nx < 20 and ny < 20:
            filename = "example_matrices/matrix_" + str(nx) + "x" + str(ny) + "x" + str(nz) + ".txt"
            np.savetxt(filename, A, fmt='%5g')

        # save some metadata to a file
        filename = "example_matrices/matrix_" + str(nz) + "x" + str(ny) + "x" + str(nz) + "_metadata.txt"
        with open(filename, 'w') as f:
            # f.write("num_nnz_per_row: " + str(num_nnz_per_row) + "\n")
            # one row per line
            for row in col_idx_per_row:
                f.write(str(row) + "\n")
            # f.write("col_idx_per_row: " + str(col_idx_per_row) + "\n")

    # print (num_zeros_before_diag)

    # print("Number of elements in the matrix: ", (nx*ny*nz)**2)
    # print("Number of non-zero elements in the inverse matrix: ", inverse_nnz)
    # print (len(num_zeros_before_z_jump_diag))
    # print (num_zeros_before_z_jump_diag)

    return A, y

def One_D_example(n):
    A = np.zeros((n, n))

    for i in range (n):
        A[i, i] = 2
        if i > 0:
            A[i, i-1] = -1
        if i < n-1:
            A[i, i+1] = -1     

    print(A)   

    # multiply the matrix with a vector of all ones
    b = np.ones(n)
    x = A @ b
    print(x)

    A_inverse = np.linalg.inv(A)

    print(A_inverse)

    print(A_inverse @ x)

# num = 4

# One_D_example(num)

# A, y = generate_Dense_Problem(num, num, num)
# generate_coarse_problem(num, num, num)