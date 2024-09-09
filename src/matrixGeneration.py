import numpy as np

def generateDenseMatrix(nx, ny, nz):

    test_simple = True
  
    # generate a np matrix with nx*ny*nz rows and columns
    A = np.zeros((nx*ny*nz, nx*ny*nz))
    num_nnz_per_row = np.zeros(nx*ny*nz)
    col_idx_per_row = []
    stair_formation_rows = []
    num_zeros_before_diag = []
    num_zeros_before_z_jump_diag = []


    # iterate over the rows of the matrix
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                col_idx_i = []
                for sz in range(-1, 2):
                    if iz+sz > -1 and iz+sz < nz:
                        for sy in range(-1, 2):
                            if iy+sy > -1 and iy+sy < ny:
                                for sx in range(-1, 2):
                                    if ix+sx > -1 and ix+sx < nx:
                                        j = ix+sx + nx*(iy+sy) + nx*ny*(iz+sz)
                                        if i == j:
                                            A[i, j] = 26.0
                                        else:
                                            A[i, j] = -1.0
                                        num_nnz_per_row[i] += 1
                                        col_idx_i.append(j)
                col_idx_per_row.append(col_idx_i)
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                if i-2 > 0:
                    if i - 1 > 0:
                        if A[i, i-1] == 0 and A[i, i-2] == 0:
                            stair_formation_rows.append(i)

                if A[i, i-1] == 0 and i > 1:
                    num_zeros = 1

                    while i - num_zeros -1 > 0 and (A[i, i-num_zeros-1] == 0):
                        num_zeros += 1
                    
                    num_zeros_before_diag.append(num_zeros)

                    if i % (nx * ny) == 0:
                        num_zeros_before_z_jump_diag.append(num_zeros)

                


                

    A = np.vstack([np.arange(2, A.shape[0] + 2), A])
    # print the matrix to a file
    if nz < 20 and nx < 20 and ny < 20:
        filename = "example_matrices/matrix_" + str(nx) + "x" + str(ny) + "x" + str(nz) + ".txt"
        np.savetxt(filename, A, fmt='%5g')

    # save some metadata to a file
    filename = "example_matrices/matrix_" + str(nz) + "x" + str(ny) + "x" + str(nz) + "_metadata.txt"
    with open(filename, 'w') as f:
        # f.write("num_nnz_per_row: " + str(num_nnz_per_row) + "\n")
        # one row per line
        for row in col_idx_per_row:
            f.write(str(row) + "\n")
        # f.write("col_idx_per_row: " + str(col_idx_per_row) + "\n")
    # print (num_zeros_before_diag)
    print (len(num_zeros_before_z_jump_diag))
    print (num_zeros_before_z_jump_diag)

num = 30

generateDenseMatrix(num, num, num)