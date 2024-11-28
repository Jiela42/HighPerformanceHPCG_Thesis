import cupy as cp


from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations as generations
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixConversions import banded_to_csr, csr_to_banded
from HighPerformanceHPCG_Thesis.Python_HPCGLib.HPCGLib_versions import BaseCuPy, NaiveBandedCuPy
from HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib import abstractHPCG_tests as abstract_tests
from HighPerformanceHPCG_Thesis.Python_HPCGLib.util import developer_mode
def test_NaiveBandedCuPy_3d27p_on_random_matrix(A_banded: BandedMatrix, A_dense: cp.ndarray) -> bool:
    all_tests_passed = True

    A_csr = CSRMatrix()
    banded_to_csr(A_banded, A_csr)

    A_sparse_cupy = A_csr.get_cupy_matrix()

    nx = A_banded.nx
    ny = A_banded.ny
    nz = A_banded.nz

    j_min_i = cp.array(A_banded.j_min_i, dtype=cp.int32)

    x = cp.zeros(A_csr.num_cols, dtype=cp.float64)
    a = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    # a = cp.ones(A_csr.num_cols, dtype=cp.float64)
    b = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    alpha, beta = cp.random.rand(1).item(), cp.random.rand(1).item()
    
    testSPMV = NaiveBandedCuPy.computeSPMV
    baselineSPMV = BaseCuPy.computeSPMV

    # now we call the abstract test methods
    SPMV_test = abstract_tests.test_SPMV(
        baselineSPMV = baselineSPMV, uutSPMV = testSPMV,
        A_csr = A_csr, A_cupy_sparse= A_sparse_cupy,
        A_banded = A_banded, A_cupy_dense=A_dense, j_min_i_cupy = j_min_i,
        x_cupy = a, y_cupy = x
        )

    if not SPMV_test:
        all_tests_passed = False
        print(f"The NaiveBandedCuPy SPMV test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The NaiveBandedCuPy SPMV test passed for size: {nx}x{ny}x{nz}", flush=True)
    
    return all_tests_passed


def test_NaiveBandedCuPy_3d27p_on_matrix(A_banded: BandedMatrix, A_dense: cp.ndarray) -> bool:
    
    all_tests_passed = True

    A_csr = CSRMatrix()
    banded_to_csr(A_banded, A_csr)

    A_sparse_cupy = A_csr.get_cupy_matrix()

    nx = A_banded.nx
    ny = A_banded.ny
    nz = A_banded.nz

    j_min_i = cp.array(A_banded.j_min_i, dtype=cp.int32)

    x = cp.zeros(A_csr.num_cols, dtype=cp.float64)
    a = cp.random.rand(A_csr.num_cols, dtype=cp.float64)
    # a = cp.ones(A_csr.num_cols, dtype=cp.float64)

    testSPMV = NaiveBandedCuPy.computeSPMV
    baselineSPMV = BaseCuPy.computeSPMV

    # now we call the abstract test methods
    SPMV_test = abstract_tests.test_SPMV(
        baselineSPMV = baselineSPMV, uutSPMV = testSPMV,
        A_csr = A_csr, A_cupy_sparse= A_sparse_cupy,
        A_banded = A_banded, A_cupy_dense=A_dense, j_min_i_cupy = j_min_i,
        x_cupy = a, y_cupy = x
        )

    if not SPMV_test:
        all_tests_passed = False
        print(f"The NaiveBandedCuPy SPMV test failed for size: {nx}x{ny}x{nz}", flush=True)
    elif developer_mode:
        print(f"The NaiveBandedCuPy SPMV test passed for size: {nx}x{ny}x{nz}", flush=True)
    
    return all_tests_passed


def test_NaiveBandedCuPy_3d27p(nx: int, ny: int, nz: int):

    all_tests_passed = True

    A_banded, A, y = generations.generate_cupy_banded_problem(nx, ny, nz)

    current_matrix_test = test_NaiveBandedCuPy_3d27p_on_matrix(A_banded, A)

    if not current_matrix_test:
        all_tests_passed = False
        print(f"The NaiveBandedCuPy test failed for size: {nx}x{ny}x{nz}", flush=True)
    

    csr_matrix, A, y = generations.generate_cupy_csr_problem(nx, ny, nz)

    csr_matrix.iterative_values = cp.random.rand(csr_matrix.num_rows, dtype=cp.float64)
    A_banded = BandedMatrix()
    csr_to_banded(csr_matrix, A_banded)
    A = A_banded.get_cupy_matrix()

    current_matrix_test = test_NaiveBandedCuPy_3d27p_on_random_matrix(A_banded, A)
    
    if not current_matrix_test:
        all_tests_passed = False
        print(f"The NaiveBandedCuPy test on iterative values failed for size: {nx}x{ny}x{nz}", flush=True)
    
    return all_tests_passed


def test_NaiveBandedCuPy(nx: int, ny: int, nz: int):

    all_tests_passed = True
    all_tests_passed = all_tests_passed and test_NaiveBandedCuPy_3d27p(nx, ny, nz)

    return all_tests_passed