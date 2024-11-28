
from typing import List
import numpy as np
import cupy as cp
import torch
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixUtils import MatrixType
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import print_elem_not_found_warnings

class BandedMatrix:
    def __init__(self: 'BandedMatrix'):
        self.matrixType = MatrixType.UNKNOWN
        self.num_cols = None
        self.num_rows = None
        self.nnz = None
        self.num_bands = None
        self.j_min_i = None
        self.values = None

        self.np_matrix = None
        self.torch_matrix = None
        self.cupy_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None
    
    def set_banded_matrix(self: 'BandedMatrix', num_cols:int, num_rows:int, num_bands:int, values: List[float]):
            
            self.matrixType = MatrixType.UNKNOWN
    
            self.num_cols = num_cols
            self.num_rows = num_rows
            self.num_bands = num_bands
            self.nnz = len([element for element in values if element != 0.0])
    
            assert len(values) == num_bands * num_rows, "Error in Matrix Initialization: values must have length num_bands * num_rows"

            self.values = values
    
            self.np_matrix = None
            self.torch_matrix = None
    
            self.nx = None
            self.ny = None
            self.nz = None
    
    def to_np(self: 'BandedMatrix') -> np.ndarray:
        
        np_matrix = np.zeros((self.num_rows, self.num_bands), dtype=np.float64)
        for i in range(self.num_rows):
            for j in range(self.num_bands):
                np_matrix[i, j] = self.values[i + j * self.num_rows]
        self.np_matrix = np_matrix
        return self.np_matrix
    
    def to_torch(self: 'BandedMatrix') -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_matrix = torch.zeros(self.num_rows, self.num_bands, device=device, dtype=torch.float64)
        for i in range(self.num_rows):
            for j in range(self.num_bands):
                torch_matrix[i, j] = self.values[i + j * self.num_rows]
        self.torch_matrix = torch_matrix
        return self.torch_matrix
    
    def to_cupy(self: 'BandedMatrix') -> np.ndarray:

        self.cupy_matrix = cp.array(self.values, dtype=cp.float64)
        return self.cupy_matrix
    
    def get_np_matrix(self: 'BandedMatrix') -> np.ndarray:
        if self.np_matrix is None:
            return self.to_np()
        return self.np_matrix
    
    def get_torch_matrix(self: 'BandedMatrix') -> torch.Tensor:
        if self.torch_matrix is None:
            return self.to_torch()
        return self.torch_matrix
    
    def get_cupy_matrix(self: 'BandedMatrix') -> np.ndarray:
        if self.cupy_matrix is None:
            return self.to_cupy()
        return self.cupy_matrix   

    def get_elem(self: 'BandedMatrix', i:int, j:int)-> float:
        for band in range(self.num_bands):
            if j == i + band:
                return self.values[i + band * self.num_rows]
        
        if developer_mode and print_elem_not_found_warnings:
            print("WARNING in BandedMatrix.get_elem: Element not found in band")
        
        return 0.0
    
    def compare_to(self: 'BandedMatrix', other: 'BandedMatrix') -> bool:
        
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
        
        if self.num_bands != other.num_bands:
            if developer_mode:
                print(f"Error: num_bands does not match: {self.num_bands} != {other.num_bands}")
            return False
        
        if self.j_min_i != other.j_min_i:
            if developer_mode:
                print(f"Error: j_min_i does not match: {self.j_min_i} != {other.j_min_i}")
            return False

        if self.values != other.values:
            if developer_mode:
                for i in range(self.num_rows):
                    for band in range(self.num_bands):
                        if self.values[i + band * self.num_rows] != other.values[i + band * self.num_rows]:
                            print(f"Error: values in row {i}, col {self.j_min_i[band] + i}, band {band} does not match: {self.values[i + band * self.num_rows]} != {other.values[i + band * self.num_rows]}")
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