
from MatrixUtils import MatrixType
from typing import List
from COOMatrix import COOMatrix
from CSRMatrix import CSRMatrix

class BandedMatrix:
    def __init__(self):
        self.matrixType = MatrixType.UNKNOWN
        self.num_cols = None
        self.num_rows = None
        self.num_bands = None
        self.j_min_i = None
        self.values = None

        self.np_matrix = None
        self.torch_matrix = None

        self.nx = None
        self.ny = None
        self.nz = None
    
    def __init__(self, num_cols:int, num_rows:int, num_bands:int, values: List[float]):
            
            self.matrixType = MatrixType.UNKNOWN
    
            self.num_cols = num_cols
            self.num_rows = num_rows
            self.num_bands = num_bands
    
            assert len(values) == num_bands * num_rows, "Error in Matrix Initialization: values must have length num_bands * num_rows"
    
            self.values = values
    
            self.np_matrix = None
            self.torch_matrix = None
    
            self.nx = None
            self.ny = None
            self.nz = None
