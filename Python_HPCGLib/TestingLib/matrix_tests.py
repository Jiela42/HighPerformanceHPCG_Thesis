from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixConversions import *
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode

import numpy as np

def test_3d27pt_matrix(nx: int, ny: int, nz: int) -> bool:

    all_tests_passed = True

    A_csr = CSRMatrix()
    A_csr.create_3d27pt_CSRMatrix(nx, ny, nz)

    A_coo = COOMatrix()
    csr_to_coo(A_csr, A_coo)

    coo_3d27pt = COOMatrix()
    coo_3d27pt.create_3d27pt_COOMatrix(nx, ny, nz)

    if not A_coo.compare_to(coo_3d27pt):
        all_tests_passed = False
        print(f"coo and csr 3d27pt matrices are different, for size {nx}x{ny}x{nz}", flush=True)

    A_back_to_csr = CSRMatrix()
    coo_to_csr(A_coo, A_back_to_csr)

    if not A_csr.compare_to(A_back_to_csr):
        all_tests_passed = False
        print(f"csr and coo conversions fail, for size {nx}x{ny}x{nz}", flush=True)

    A_coo.create_3d27pt_COOMatrix(nx, ny, nz)

    A_banded = BandedMatrix()
    coo_to_banded(A_coo, A_banded)

    A_back_to_coo = COOMatrix()
    banded_to_coo(A_banded, A_back_to_coo)

    if not A_coo.compare_to(A_back_to_coo):
        all_tests_passed = False
        print(f"banded to coo conversion failed for size {nx}x{ny}x{nz}", flush=True)

    if developer_mode and all_tests_passed:
        print(f"3D27p Matrix tests passed for size {nx}x{ny}x{nz}", flush=True)

    return all_tests_passed

# this function needs to get adjusted for when we also have other matrix types
def run_matrix_tests(nx: int, ny: int, nz: int) -> bool:

    all_tests_passed = True

    all_tests_passed = all_tests_passed and test_3d27pt_matrix(nx, ny, nz)

    return all_tests_passed

