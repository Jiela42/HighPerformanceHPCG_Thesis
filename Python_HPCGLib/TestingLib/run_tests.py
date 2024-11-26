import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.matrix_tests as matrix_tests
import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.BaseTorch_tests as BaseTorch_tests
import HighPerformanceHPCG_Thesis.Python_HPCGLib.TestingLib.BaseCuPy_tests as BaseCuPy_tests

print("*************************************RUNNING TEST SUITE*************************************", flush=True)

all_tests_passed = True

print("Running matrix tests", flush=True)
# all_tests_passed = all_tests_passed and matrix_tests.run_matrix_tests(3, 3, 3)
# all_tests_passed = all_tests_passed and matrix_tests.run_matrix_tests(8, 8, 8)
# all_tests_passed = all_tests_passed and matrix_tests.run_matrix_tests(16, 16, 16)
# all_tests_passed = all_tests_passed and matrix_tests.run_matrix_tests(32, 32, 32)
# all_tests_passed = all_tests_passed and matrix_tests.run_matrix_tests(64, 64, 64)
print("Matrix tests done", flush=True)

print("Running HPCG tests", flush=True)
# all_tests_passed = all_tests_passed and BaseTorch_tests.test_BaseTorch(8, 8, 8)
# all_tests_passed = all_tests_passed and BaseTorch_tests.test_BaseTorch(16, 16, 16)
# all_tests_passed = all_tests_passed and BaseTorch_tests.test_BaseTorch(24, 24, 24)
# all_tests_passed = all_tests_passed and BaseTorch_tests.test_BaseTorch(32, 32, 32)
# all_tests_passed = all_tests_passed and BaseTorch_tests.test_BaseTorch(64, 64, 64)
# all_tests_passed = all_tests_passed and BaseTorch_tests.test_BaseTorch(128, 128, 128)
print("HPCG tests done", flush=True)

print("Starting BaseCuPy tests", flush=True)

all_tests_passed = all_tests_passed and BaseCuPy_tests.test_BaseCuPy(8, 8, 8)
all_tests_passed = all_tests_passed and BaseCuPy_tests.test_BaseCuPy(16, 16, 16)
all_tests_passed = all_tests_passed and BaseCuPy_tests.test_BaseCuPy(24, 24, 24)
all_tests_passed = all_tests_passed and BaseCuPy_tests.test_BaseCuPy(32, 32, 32)
all_tests_passed = all_tests_passed and BaseCuPy_tests.test_BaseCuPy(64, 64, 64)
# all_tests_passed = all_tests_passed and BaseCuPy_tests.test_BaseCuPy(128, 128, 128)

print("BaseCuPy tests done", flush=True)


if all_tests_passed:
    print("********************************************************************************************", flush=True)
    print("**************************************ALL TESTS PASSED**************************************", flush=True)
    print("********************************************************************************************", flush=True)
else:
    print("some tests failed -> go do debugging", flush=True)

# print(A)