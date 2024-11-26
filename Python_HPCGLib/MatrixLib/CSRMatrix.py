import numpy as np
import torch
import cupy as cp
import cupyx.scipy.sparse as sp

from typing import List, Tuple
import itertools

from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixUtils import MatrixType
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import print_elem_not_found_warnings

class CSRMatrix:

    def __init__(self: 'CSRMatrix'):
        self.matrixType = MatrixType.UNKNOWN
        self.num_cols = None
        self.num_rows = None
        self.nnz = None
        self.row_ptr = None
        self.col_idx =  None
        self.values = None

        self.np_matrix = None
        self.torch_matrix = None
        self.cupy_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None

    def set_CSRMatrix(self: 'CSRMatrix', num_cols:int, num_rows:int, row_ptr: List[int], col_idx: List[int], values: List[float]):
        
        self.matrixType = MatrixType.UNKNOWN

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.nnz = len(values)

        assert len(row_ptr) == num_rows + 1, "Error in Matrix Initialization: row_ptr must have length num_rows + 1"
        assert len(col_idx) == values, "Error in Matrix Initialization: col_idx must have length values"
        assert num_cols > max(col_idx), "Error in Matrix Initialization: num_cols must be greater than the maximum value in col_idx"

        self.row_ptr = row_ptr
        self.col_idx = col_idx
        self.values = values

        self.np_matrix = None
        self.torch_matrix = None
        self.cupy_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None

    def create_3d27pt_CSRMatrix(self: 'CSRMatrix', nx: int, ny: int, nz: int):
        self.matrixType = MatrixType.Stencil_3D27P
        self.num_rows = nx * ny * nz
        self.num_cols = nx * ny * nz
        self.nx = nx
        self.ny = ny
        self.nz = nz

        num_rows = self.num_rows

        nnz = 0
        col_indices = [[] for _ in range(num_rows)]
        values = [[] for _ in range(num_rows)]

        # This will be the csr row index array which is why we prepend a zero
        # (we get this by doing a prefix sum later ;))
        nnz_per_row = [0 for _ in range(num_rows+1)]

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
                                            j = ix+sx + nx*(iy+sy) + nx*ny*(iz+sz)
                                            if i == j:
                                                col_indices[i].append(j)
                                                values[i].append(26.0)
                                            else:
                                                col_indices[i].append(j)
                                                values[i].append(-1.0)
                                            nnz_i += 1
                    nnz_per_row[i] = nnz_i
                    nnz += nnz_i

        # Concatenate all lists in col_indices and values
        col_indices = list(itertools.chain(*col_indices))
        values = list(itertools.chain(*values))

        row_idx = [0]

        for i in range(num_rows):
            # this works because we are appending and because we prepended a zero
            # this means for row i, the row_idx[i] contains the sum of nnz for all previous rows, not including row i
            row_idx.append(row_idx[i] + nnz_per_row[i])

        self.nnz = nnz
        self.row_ptr = row_idx
        self.col_idx = col_indices
        self.values = values

    def iterative_values(self: 'CSRMatrix'):
        next_val = 0.1
        for i in range(self.nnz):
            self.values[i] = next_val
            next_val += 0.1
            if next_val > 11.1:
                next_val = 0.1

    def to_np(self: 'CSRMatrix') -> Tuple[np.array, np.array, np.array]:

       
        row_ptr_np = np.array(self.row_ptr, dtype=np.int32)
        col_idx_np = np.array(self.col_idx, dtype=np.int32)
        values_np = np.array(self.values, dtype=np.float64)

        self.np_matrix = (row_ptr_np, col_idx_np, values_np)

        return self.np_matrix

    def to_torch(self: 'CSRMatrix') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # note that this version of torch (which we have to use for the GPU we work on),
        # does not have CSR matrix support so we just return three tensors

        row_ptr_torch = torch.tensor(self.row_ptr, device=device, dtype=torch.int32)
        col_idx_torch = torch.tensor(self.col_idx, device=device, dtype=torch.int32)
        values_torch = torch.tensor(self.values, device=device, dtype=torch.float64)

        self.torch_matrix = (row_ptr_torch, col_idx_torch, values_torch)

        return self.torch_matrix

    def to_cupy(self: 'CSRMatrix') -> sp.csr_matrix:
        
        row_ptr_np, col_idx_np, values_np = self.to_np()
        row_ptr_cp = cp.array(row_ptr_np)
        col_idx_cp = cp.array(col_idx_np)
        values_cp = cp.array(values_np)

        self.cupy_matrix = None
        self.cupy_matrix = sp.csr_matrix((values_cp, col_idx_cp, row_ptr_cp), shape=(self.num_rows, self.num_cols))

        return self.cupy_matrix

    def get_np_matrix(self: 'CSRMatrix') -> Tuple[np.array, np.array, np.array]:
        if self.np_matrix is None:
            return self.to_np()
        return self.np_matrix
    
    def get_torch_matrix(self: 'CSRMatrix') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        if self.torch_matrix is None:
            return self.to_torch()
        return self.torch_matrix
    
    def get_cupy_matrix(self: 'CSRMatrix') -> sp.csr_matrix:
        if self.cupy_matrix is None:
            return self.to_cupy()
        return self.cupy_matrix

    def get_element(self: 'CSRMatrix', i: int, j: int) -> float:
        for k in range(self.row_ptr[i], self.row_ptr[i+1]):
            if self.col_idx[k] == j:
                return self.values[k]
        if developer_mode and print_elem_not_found_warnings:
            print(f"WARNING: Element ({i}, {j}) not found in CSR matrix")
        return 0.0

    def compare_to(self: 'CSRMatrix', other: 'CSRMatrix') -> bool:
        if self.num_cols != other.num_cols:
            if developer_mode:
                print(f"Error: num_cols does not match: {self.num_cols} != {other.num_cols}")
            return False
        
        if self.num_rows != other.num_rows:
            if developer_mode:
                print(f"Error: num_rows does not match: {self.num_rows} != {other.num_rows}")
            return False
        
        if self.nnz != other.nnz:
            if developer_mode:
                print(f"Error: nnz does not match: {self.nnz} != {other.nnz}")
            return False
        
        if self.row_ptr != other.row_ptr:
            if developer_mode:
                print(f"Error: row_ptr does not match: {self.row_ptr} != {other.row_ptr}")
            return False
        
        if self.col_idx != other.col_idx:
            if developer_mode:
                print(f"Error: col_idx does not match: {self.col_idx} != {other.col_idx}")
            return False
        
        if self.values != other.values:
            if developer_mode:
                for i in range(self.num_rows):
                    for j in range(self.row_ptr[i], self.row_ptr[i+1]):
                        if self.values[j] != other.values[j]:
                            print(f"Error: values at row {i}, col {self.col_idx[j]} do not match: {self.values[j]} != {other.values[j]}")
                            break
            return False
        
        if self.nx != other.nx:
            if developer_mode:
                print(f"Error: nx does not match: {self.nx} != {other.nx}")
            return False
        
        if self.ny != other.ny:
            if developer_mode:
                print(f"Error: ny does not match: {self.ny} != {other.ny}")
            return False
        
        if self.nz != other.nz:
            if developer_mode:
                print(f"Error: nz does not match: {self.nz} != {other.nz}")
            return False
        
        if self.matrixType != other.matrixType:
            if developer_mode:
                print(f"Error: matrixType does not match: {self.matrixType} != {other.matrixType}")
            return False
        return True
    
    def to_dense(self: 'CSRMatrix') -> np.array:
        dense_matrix = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.row_ptr[i], self.row_ptr[i+1]):
                dense_matrix[i, self.col_idx[j]] = self.values[j]
        return dense_matrix
    