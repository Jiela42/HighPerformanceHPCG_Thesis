import numpy as np
import torch

from typing import List, Tuple
import itertools

from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixUtils import MatrixType
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import print_elem_not_found_warnings

class COOMatrix:

    def __init__(self: 'COOMatrix'):
        self.matrixType = MatrixType.UNKNOWN
        self.num_cols = None
        self.num_rows = None
        self.nnz = None
        self.row_idx = None
        self.col_idx =  None
        self.values = None

        self.np_matrix = None
        self.torch_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None


    def set_COOMatrix(self: 'COOMatrix', num_cols:int, num_rows:int, row_idx: List[int], col_idx: List[int], values: List[float]):
        
        self.matrixType = MatrixType.UNKNOWN

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.nnz = len(values)

        assert len(row_idx) == len(col_idx), "Error in Matrix Initialization: row_ptr must have length num_rows + 1"
        assert len(col_idx) == len(values), "Error in Matrix Initialization: col_idx must have length values"
        assert num_cols > max(col_idx), "Error in Matrix Initialization: num_cols must be greater than the maximum value in col_idx"
        assert num_rows > max(row_idx), "Error in Matrix Initialization: num_rows must be greater than the maximum value in row_idx"

        self.row_idx = row_idx
        self.col_idx = col_idx
        self.values = values

        self.np_matrix = None
        self.torch_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None

    def create_3d27pt_COOMatrix(self: 'COOMatrix', nx: int, ny: int, nz: int):
        self.matrixType = MatrixType.Stencil_3D27P
        self.num_rows = nx * ny * nz
        self.num_cols = nx * ny * nz
        self.nx = nx
        self.ny = ny
        self.nz = nz

        row_col_val_tuples = []
        nnz = 0

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
                                                row_col_val_tuples.append((i, j, 26.0))
                                                nnz += 1
                                            else:
                                                row_col_val_tuples.append((i, j, -1.0))
                                                nnz += 1

        # sort the tuples by row index and then by column index
        row_col_val_tuples.sort(key=lambda x: (x[0], x[1]))

        row_idx = []
        col_indices = []
        values = []

        for row, col, val in row_col_val_tuples:
            row_idx.append(row)
            col_indices.append(col)
            values.append(val)

        self.nnz = nnz
        self.row_idx = row_idx
        self.col_idx = col_indices
        self.values = values

    def iterative_values(self: 'COOMatrix'):
        next_val = 0.1
        for i in range(self.nnz):
            self.values[i] = next_val
            next_val += 0.1
            if next_val > 11.1:
                next_val = 0.1

    def to_np(self: 'COOMatrix') -> Tuple[np.array, np.array, np.array]:

        # note that np does not support COO Matrices, so we return np arrays
        if self.np_matrix is not None:
            return self.np_matrix
        else:
            row_idx_np = np.array(self.row_ptr, dtype=np.int32)
            col_idx_np = np.array(self.col_idx, dtype=np.int32)
            values_np = np.array(self.values, dtype=np.float64)

            self.np_matrix = (row_idx_np, col_idx_np, values_np)

            return self.np_matrix

    def to_torch(self: 'COOMatrix') -> torch.sparse.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        row_ptr_torch = torch.tensor(self.row_idx, device=device, dtype=torch.int32)
        col_idx_torch = torch.tensor(self.col_idx, device=device, dtype=torch.int32)
        values_torch = torch.tensor(self.values, device=device, dtype=torch.float64)

        A = torch.sparse_coo_tensor(torch.stack([row_ptr_torch, col_idx_torch]), values_torch, (self.num_rows, self.num_cols), device=device, dtype=torch.float64)
        self.torch_matrix = A.coalesce()

        return self.torch_matrix

    def get_torch_matrix(self: 'COOMatrix') -> torch.sparse.Tensor:
        if self.torch_matrix is not None:
            return self.torch_matrix
        else:
            return self.to_torch()

    def get_element(self: 'COOMatrix', i:int, j:int)->float:
        
        for k in range(len(self.values)):
            if self.row_idx[k] == i and self.col_idx[k] == j:
                return self.values[k]
        
        if developer_mode and print_elem_not_found_warnings:
            print(f"WARNING: Element ({i}, {j}) not found in COO matrix")
        return 0.0
    
    def compare_to(self: 'COOMatrix', other: 'COOMatrix') -> bool:

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
        
        if self.row_idx != other.row_idx:
            if developer_mode:
                print(f"Error: row_idx does not match: {self.row_idx} != {other.row_idx}")
            return False
        
        if self.col_idx != other.col_idx:
            if developer_mode:
                print(f"Error: col_idx does not match: {self.col_idx} != {other.col_idx}")
            return False
        
        if self.values != other.values:
            if developer_mode:
                for i in range(len(self.values)):
                    if self.values[i] != other.values[i]:
                        print(f"Error: values at row {self.row_idx[i]}, col {self.col_idx[i]} do not match: {self.values[i]} != {other.values[i]}")
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
        
        # these may not have been created by one or the other
        # if self.np_matrix != other.np_matrix:
        #     if developer_mode:
        #         print(f"Error: np_matrix does not match")
        #     return False
        
        # if self.torch_matrix != other.torch_matrix:
        #     if developer_mode:
        #         print(f"Error: torch_matrix does not match")
        #     return False   

        return True

    def to_dense(self: 'COOMatrix') -> np.array:
        dense_matrix = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.nnz):
            dense_matrix[self.row_idx[i], self.col_idx[i]] = self.values[i]
        return dense_matrix
    