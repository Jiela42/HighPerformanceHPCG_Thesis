
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixUtils import MatrixType
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.COOMatrix import COOMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.BandedMatrix import BandedMatrix
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.CSRMatrix import CSRMatrix
# from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib.MatrixUtils import developer_mode



def csr_to_coo(csrMatrix: CSRMatrix, cooMatrix: COOMatrix):
    cooMatrix.matrixType = csrMatrix.matrixType
    cooMatrix.nx = csrMatrix.nx
    cooMatrix.ny = csrMatrix.ny
    cooMatrix.nz = csrMatrix.nz
    cooMatrix.num_cols = csrMatrix.num_cols
    cooMatrix.num_rows = csrMatrix.num_rows
    cooMatrix.nnz = csrMatrix.nnz

    coo_row_idx = []
    coo_col_idx = []
    # loop_vals = []
    coo_values = csrMatrix.values.copy()

    for i in range(csrMatrix.num_rows):
        for j in range(csrMatrix.row_ptr[i], csrMatrix.row_ptr[i+1]):
            coo_row_idx.append(i)
            coo_col_idx.append(csrMatrix.col_idx[j])
            # loop_vals.append(csrMatrix.values[j])

    # print(f"csr row_ptr: {csrMatrix.row_ptr}")

    # print(f"coo row_idx: {coo_row_idx}")
    # print(f"csr col_idx: {csrMatrix.col_idx}")
    # print(f"coo col_idx: {coo_col_idx}")       

    # print(f"nnz: {csrMatrix.nnz}")
    # print(f"len(coo_row_idx): {len(coo_row_idx)}")
    # print(f"len(coo_col_idx): {len(coo_col_idx)}")
    # print(f"len(coo_values): {len(coo_values)}")

    # print(f"loop_vals: {loop_vals}")
    # print(f"coo_values: {coo_values}")
    assert len(coo_row_idx) == len(coo_col_idx) == len(coo_values), "Error in Matrix Conversion: row_idx, col_idx, and values must have the same length"

    cooMatrix.row_idx = coo_row_idx
    cooMatrix.col_idx = coo_col_idx
    cooMatrix.values = coo_values

def csr_to_banded(csrMatrix: CSRMatrix, bandedMatrix: BandedMatrix):
    if csrMatrix.matrixType == MatrixType.Stencil_3D27P:
        bandedMatrix.matrixType = csrMatrix.matrixType
        bandedMatrix.nx = csrMatrix.nx
        bandedMatrix.ny = csrMatrix.ny
        bandedMatrix.nz = csrMatrix.nz
        bandedMatrix.nnz = csrMatrix.nnz
        bandedMatrix.num_cols = csrMatrix.num_cols
        bandedMatrix.num_rows = csrMatrix.num_rows
        bandedMatrix.num_bands = 27

        banded_values = [[0.0 for _ in range(csrMatrix.num_bands)] for _ in range(csrMatrix.num_rows)]
        j_min_i = []
        neighbour_offsets =[
            (-1, -1, -1), (0, -1, -1), (1, -1, -1),
            (-1, 0, -1), (0, 0, -1), (1, 0, -1),
            (-1, 1, -1), (0, 1, -1), (1, 1, -1),
            (-1, -1, 0), (0, -1, 0), (1, -1, 0),
            (-1, 0, 0), (0, 0, 0), (1, 0, 0),
            (-1, 1, 0), (0, 1, 0), (1, 1, 0),
            (-1, -1, 1), (0, -1, 1), (1, -1, 1),
            (-1, 0, 1), (0, 0, 1), (1, 0, 1),
            (-1, 1, 1), (0, 1, 1), (1, 1, 1)
        ]

        for i in range(csrMatrix.num_bands):
            off_x, off_y, off_z = neighbour_offsets[i]
            j_min_i.append(off_x + csrMatrix.nx*off_y + csrMatrix.nx*csrMatrix.ny*off_z)

        nnz_ctr = 0
        for i in range(csrMatrix.num_rows):
            for band_j in range(csrMatrix.num_bands):
                j = j_min_i[band_j] + i
                # check if j is in bounds (since not every point has all 27 neighbours)
                if (j >= 0 and j < csrMatrix.num_cols):
                    elem = csrMatrix.get_element(i, j)
                    #  also make sure we don't add zero elements
                    if elem != 0.0:
                        banded_values[i * csrMatrix.num_bands + band_j] = elem
                        nnz_ctr += 1
        
        assert nnz_ctr == csrMatrix.nnz, "Error: Number of non-zero elements in banded matrix does not match number of non-zero elements in CSR matrix"

    else:
        assert False, "Error: to_banded only implemented for Stencil_3D27P matrices"

     
def banded_to_csr(bandedMatrix: BandedMatrix, csrMatrix: CSRMatrix):
    csrMatrix.num_cols = bandedMatrix.num_cols
    csrMatrix.num_rows = bandedMatrix.num_rows
    csrMatrix.nnz = bandedMatrix.nnz
    csrMatrix.matrixType = bandedMatrix.matrixType
    csrMatrix.nx = bandedMatrix.nx
    csrMatrix.ny = bandedMatrix.ny
    csrMatrix.nz = bandedMatrix.nz

    csr_row_ptr = [0 for _ in range(bandedMatrix.num_rows + 1)]
    csr_col_ind = [0 for _ in range(bandedMatrix.nnz)]
    csr_values = [0.0 for _ in range(bandedMatrix.nnz)]

    elem_ctr = 0

    for i in range(bandedMatrix.num_rows):
        for band_j in range(bandedMatrix.num_bands):
            j = bandedMatrix.j_min_i[band_j] + i
            val = bandedMatrix.values[i * bandedMatrix.num_rows + band_j]
            if val != 0.0:
                csr_col_ind[elem_ctr] = j
                csr_values[elem_ctr] = val
                elem_ctr += 1
        csr_row_ptr[i + 1] = elem_ctr

    assert elem_ctr == bandedMatrix.nnz, f"Error in BandedMatrix.to_CSRMatrix: elem_ctr != bandedMatrix.nnz, {elem_ctr} != {bandedMatrix.nnz}"

def banded_to_coo(bandedMatrix: BandedMatrix, cooMatrix: COOMatrix):
    cooMatrix.num_cols = bandedMatrix.num_cols
    cooMatrix.num_rows = bandedMatrix.num_rows
    cooMatrix.nnz = bandedMatrix.nnz
    cooMatrix.matrixType = bandedMatrix.matrixType
    cooMatrix.nx = bandedMatrix.nx
    cooMatrix.ny = bandedMatrix.ny
    cooMatrix.nz = bandedMatrix.nz

    coo_row_ind = [0 for _ in range(bandedMatrix.nnz)]
    coo_col_ind = [0 for _ in range(bandedMatrix.nnz)]
    coo_values = [0.0 for _ in range(bandedMatrix.nnz)]

    elem_ctr = 0

    for i in range(bandedMatrix.num_rows):
        for band_j in range(bandedMatrix.num_bands):
            j = bandedMatrix.j_min_i[band_j] + i
            val = bandedMatrix.values[i * bandedMatrix.num_rows + band_j]
            if val != 0.0:
                coo_row_ind[elem_ctr] = i
                coo_col_ind[elem_ctr] = j
                coo_values[elem_ctr] = val
                elem_ctr += 1

    assert elem_ctr == bandedMatrix.nnz, f"Error in BandedMatrix.to_COOMatrix: elem_ctr != bandedMatrix.nnz, {elem_ctr} != {bandedMatrix.nnz}"


def coo_to_csr(cooMatrix: COOMatrix, csrMatrix: CSRMatrix):

    # check that the coo matrix is sorted
    assert all(cooMatrix.row_idx[i] <= cooMatrix.row_idx[i+1] for i in range(len(cooMatrix.row_idx)-1)), "Error in Matrix Conversion: row_idx must be sorted"

    assert len(cooMatrix.row_idx) == len(cooMatrix.col_idx) == len(cooMatrix.values), "Error in Matrix Conversion: row_idx, col_idx, and values must have the same length"

    csrMatrix.matrixType = cooMatrix.matrixType
    csrMatrix.num_cols = cooMatrix.num_cols
    csrMatrix.num_rows = cooMatrix.num_rows
    csrMatrix.nnz = cooMatrix.nnz
    # csrMatrix.row_idx = cooMatrix.row_idx
    # csrMatrix.col_idx = cooMatrix.col_idx
    # csrMatrix.values = cooMatrix.values

    csrMatrix.nx = cooMatrix.nx
    csrMatrix.ny = cooMatrix.ny
    csrMatrix.nz = cooMatrix.nz

    nnz_per_row = [0 for _ in range(cooMatrix.num_rows)]
    row_ptr_csr = [0]
    col_idx_csr = cooMatrix.col_idx.copy()
    values_csr = cooMatrix.values.copy()

    for i in range(len(cooMatrix.values)):
        row = cooMatrix.row_idx[i]
        nnz_per_row[row] += 1
    
    for i in range(cooMatrix.num_rows):
        row_ptr_csr.append(row_ptr_csr[i] + nnz_per_row[i])

    csrMatrix.row_ptr = row_ptr_csr
    csrMatrix.col_idx = col_idx_csr
    csrMatrix.values = values_csr

def coo_to_banded(cooMatrix: COOMatrix, bandedMatrix: BandedMatrix):
    if cooMatrix.matrixType == MatrixType.Stencil_3D27P:
        bandedMatrix.num_cols = cooMatrix.num_cols
        bandedMatrix.num_rows = cooMatrix.num_rows
        bandedMatrix.num_bands = 27

        banded_values = [[0.0 for _ in range(cooMatrix.num_bands)] for _ in range(cooMatrix.num_rows)]
        j_min_i = []
        neighbour_offsets =[
            (-1, -1, -1), (0, -1, -1), (1, -1, -1),
            (-1, 0, -1), (0, 0, -1), (1, 0, -1),
            (-1, 1, -1), (0, 1, -1), (1, 1, -1),
            (-1, -1, 0), (0, -1, 0), (1, -1, 0),
            (-1, 0, 0), (0, 0, 0), (1, 0, 0),
            (-1, 1, 0), (0, 1, 0), (1, 1, 0),
            (-1, -1, 1), (0, -1, 1), (1, -1, 1),
            (-1, 0, 1), (0, 0, 1), (1, 0, 1),
            (-1, 1, 1), (0, 1, 1), (1, 1, 1)
        ]

        for i in range(cooMatrix.num_bands):
            off_x, off_y, off_z = neighbour_offsets[i]
            j_min_i.append(off_x + cooMatrix.nx*off_y + cooMatrix.nx*cooMatrix.ny*off_z)

        nnz_ctr = 0

        for elem in range(len(cooMatrix.values)):
            i = cooMatrix.row_idx[elem]
            j = cooMatrix.col_idx[elem]
            elem = cooMatrix.values[elem]
            #  make sure we don't add zero elements
            if elem != 0.0:
                # look for the right band to store the element in
                for band in range(cooMatrix.num_bands):
                    if j - i == j_min_i[band]:
                        banded_values[i][band] = elem
                        nnz_ctr += 1
        
        assert nnz_ctr == cooMatrix.nnz, "Error: Number of non-zero elements in banded matrix does not match number of non-zero elements in CSR matrix"

    else:
        assert False, "Error: to_banded only implemented for Stencil_3D27P matrices"


