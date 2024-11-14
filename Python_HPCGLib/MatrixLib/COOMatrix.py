import numpy as np
import torch

from typing import List, Tuple
import itertools

from MatrixUtils import MatrixType
from CSRMatrix import CSRMatrix


class CSRMatrix:

    def __init__(self):
        self.matrixType = MatrixType.UNKNOWN
        self.num_cols = None
        self.num_rows = None
        self.row_idx = None
        self.col_idx =  None
        self.values = None

        self.np_matrix = None
        self.torch_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None


    def __init__(self, num_cols:int, num_rows:int, row_idx: List[int], col_idx: List[int], values: List[float]):
        
        self.matrixType = MatrixType.UNKNOWN

        self.num_cols = num_cols
        self.num_rows = num_rows

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

    def create_3d27pt_COOMatrix(self, nx: int, ny: int, nz: int):
        self.matrixType = MatrixType.Stencil_3D27P
        self.num_rows = nx * ny * nz
        self.num_cols = nx * ny * nz
        self.nx = nx
        self.ny = ny
        self.nz = nz

        num_rows = self.num_rows

        row_idx = []
        col_indices = []
        values = []


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
                                                row_idx.append(i)
                                                col_indices.append(j)
                                                values.append(26.0)
                                            else:
                                                row_idx.append(i)
                                                col_indices.append(j)
                                                values.append(-1.0)

        self.row_idx = row_idx
        self.col_idx = col_indices
        self.values = values

    def to_np(self) -> Tuple[np.array, np.array, np.array]:

        # note that np does not support COO Matrices, so we return np arrays
        if self.np_matrix is not None:
            return self.np_matrix
        else:
            row_idx_np = np.array(self.row_ptr, dtype=np.int32)
            col_idx_np = np.array(self.col_idx, dtype=np.int32)
            values_np = np.array(self.values, dtype=np.float64)

            self.np_matrix = (row_idx_np, col_idx_np, values_np)

            return self.np_matrix

    def to_torch(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # note that this version of torch (which we have to use for the GPU we work on),
        # does not have CSR matrix support so we just return three tensors

        if self.torch_matrix is not None:
            return self.torch_matrix
        else:
            row_ptr_torch = torch.tensor(self.row_idx, device=device, dtype=torch.int32)
            col_idx_torch = torch.tensor(self.col_idx, device=device, dtype=torch.int32)
            values_torch = torch.tensor(self.values, device=device, dtype=torch.float64)

            A = torch.sparse_coo_tensor(torch.stack([row_ptr_torch, col_idx_torch]), values_torch, (self.num_rows, self.num_cols), device=device, dtype=torch.float64)
            self.torch_matrix = A.coalesce()

            return self.torch_matrix


    def to_csr(self, csrMatrix: CSRMatrix):

        # check that the coo matrix is sorted
        assert all(self.row_idx[i] <= self.row_idx[i+1] for i in range(len(self.row_idx)-1)), "Error in Matrix Conversion: row_idx must be sorted"

        csrMatrix.matrixType = self.matrixType
        csrMatrix.num_cols = self.num_cols
        csrMatrix.num_rows = self.num_rows
        csrMatrix.row_idx = self.row_idx
        csrMatrix.col_idx = self.col_idx
        csrMatrix.values = self.values

        csrMatrix.nx = self.nx
        csrMatrix.ny = self.ny
        csrMatrix.nz = self.nz

        row_ptr_csr = [0 for _ in range(self.num_rows+1)]
        col_idx_csr = self.col_idx.copy()
        values_csr = self.values.copy()

        for i in range(len(self.values)):
            row_ptr_csr[self.row_idx[i+1]] += 1

        csrMatrix.row_ptr = row_ptr_csr
        csrMatrix.col_idx = col_idx_csr
        csrMatrix.values = values_csr

    # def to_banded(self):


    