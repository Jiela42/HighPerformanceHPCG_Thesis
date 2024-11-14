import numpy as np
import torch

from typing import List, Tuple
import itertools

from MatrixUtils import MatrixType
from COOMatrix import COOMatrix
from BandedMatrix import BandedMatrix


class CSRMatrix:

    def __init__(self):
        self.matrixType = MatrixType.UNKNOWN
        self.num_cols = None
        self.num_rows = None
        self.row_ptr = None
        self.col_idx =  None
        self.values = None

        self.np_matrix = None
        self.torch_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None


    def __init__(self, num_cols:int, num_rows:int, row_ptr: List[int], col_idx: List[int], values: List[float]):
        
        self.matrixType = MatrixType.UNKNOWN

        self.num_cols = num_cols
        self.num_rows = num_rows

        assert len(row_ptr) == num_rows + 1, "Error in Matrix Initialization: row_ptr must have length num_rows + 1"
        assert len(col_idx) == values, "Error in Matrix Initialization: col_idx must have length values"
        assert num_cols > max(col_idx), "Error in Matrix Initialization: num_cols must be greater than the maximum value in col_idx"

        self.row_ptr = row_ptr
        self.col_idx = col_idx
        self.values = values

        self.np_matrix = None
        self.torch_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None

    def create_3d27pt_CSRMatrix(self, nx: int, ny: int, nz: int):
        self.matrixType = MatrixType.Stencil_3D27P
        self.num_rows = nx * ny * nz
        self.num_cols = nx * ny * nz
        self.nx = nx
        self.ny = ny
        self.nz = nz

        num_rows = self.num_rows


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

        # Concatenate all lists in col_indices and values
        col_indices = list(itertools.chain(*col_indices))
        values = list(itertools.chain(*values))

        row_idx = [0]

        for i in range(1,num_rows+1):
            row_idx.append(row_idx[i-1] + nnz_per_row[i])

        self.row_ptr = row_idx
        self.col_idx = col_indices
        self.values = values

    def to_np(self) -> Tuple[np.array, np.array, np.array]:

        # note that np does not support CSR Matrices, so we return np arrays
        if self.np_matrix is not None:
            return self.np_matrix
        else:
            row_ptr_np = np.array(self.row_ptr, dtype=np.int32)
            col_idx_np = np.array(self.col_idx, dtype=np.int32)
            values_np = np.array(self.values, dtype=np.float64)

            self.np_matrix = (row_ptr_np, col_idx_np, values_np)

            return self.np_matrix

    def to_torch(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # note that this version of torch (which we have to use for the GPU we work on),
        # does not have CSR matrix support so we just return three tensors

        if self.torch_matrix is not None:
            return self.torch_matrix
        else:
            row_ptr_torch = torch.tensor(self.row_ptr, device=device, dtype=torch.int32)
            col_idx_torch = torch.tensor(self.col_idx, device=device, dtype=torch.int32)
            values_torch = torch.tensor(self.values, device=device, dtype=torch.float64)

            self.torch_matrix = (row_ptr_torch, col_idx_torch, values_torch)

            return self.torch_matrix


    def to_coo(self, cooMatrix: COOMatrix):
        cooMatrix.num_cols = self.num_cols
        cooMatrix.num_rows = self.num_rows

        coo_row_idx = []
        coo_col_idx = []
        coo_values = self.values.copy()

        for i in range(self.num_rows):
            for j in range(self.row_ptr[i], self.row_ptr[i+1]):
                coo_row_idx.append(i)
                coo_col_idx.append(self.col_idx[j])

        cooMatrix.row_idx = coo_row_idx
        cooMatrix.col_idx = coo_col_idx
        cooMatrix.values = coo_values


    def to_banded(self, bandedMatrix: BandedMatrix):
        if self.matrixType == Stencil_3D27P:
            bandedMatrix.num_cols = self.num_cols
            bandedMatrix.num_rows = self.num_rows
            bandedMatrix.num_bands = 27

            banded_values = [[]]
            j_min_i =[      [(-1, -1, -1): 0, (0, -1, -1): 1, (1, -1, -1): 2,
        (-1, 0, -1): 3, (0, 0, -1): 4, (1, 0, -1): 5,
        (-1, 1, -1): 6, (0, 1, -1): 7, (1, 1, -1): 8,
        (-1, -1, 0): 9, (0, -1, 0): 10, (1, -1, 0): 11,
        (-1, 0, 0): 12, (0, 0, 0): 13, (1, 0, 0): 14,
        (-1, 1, 0): 15, (0, 1, 0): 16, (1, 1, 0): 17,
        (-1, -1, 1): 18, (0, -1, 1): 19, (1, -1, 1): 20,
        (-1, 0, 1): 21, (0, 0, 1): 22, (1, 0, 1): 23,
        (-1, 1, 1): 24, (0, 1, 1): 25, (1, 1, 1): 26
    ]

        else:
            assert False, "Error: to_banded only implemented for Stencil_3D27P matrices"


    