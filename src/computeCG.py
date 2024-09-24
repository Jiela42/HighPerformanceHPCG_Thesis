import torch

# my personal implementation of generations for the matrices, vectors etc.
import generations

num_pre_smoother_steps = 1
num_post_smoother_steps = 1

def computeSPMV_stencil(nx: int, ny: int, nz: int, y: torch.tensor, x: torch.tensor) -> int:

    # iterate over the rows of the matrix
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                col_idx_i = []
                # iterate over the neighbours
                for sz in range(-1, 2):
                    if iz+sz > -1 and iz+sz < nz:
                        for sy in range(-1, 2):
                            if iy+sy > -1 and iy+sy < ny:
                                for sx in range(-1, 2):
                                    if ix+sx > -1 and ix+sx < nx:
                                        j = ix+sx + nx*(iy+sy) + nx*ny*(iz+sz)
                                        if i == j:
                                            # Ax = y
                                            y[i] += 26.0 * x[j]
                                        else:
                                            y[i] -= 1.0 * x[j]
    return 0

def computeSYMGS(nx: int, nz: int, ny: int,
                 A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor)-> int:
    
    n_rows = nx * ny * nz
    indices = A.indices()
    values = A.values()

    # forward pass
    for i in range(n_rows):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]


        diag = (nnz_values[nnz_cols == i]).item()

        sum = r[i]
        sum -= torch.dot(nnz_values, x[nnz_cols])


        sum += diag * x[i]
        x[i] = sum / diag
    
    # backward pass
    for i in range(n_rows-1, -1, -1):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]

        diag = (nnz_values[nnz_cols == i]).item()

        sum = r[i]
        sum -= torch.dot(nnz_values, x[nnz_cols])

        sum += diag * x[i]
        x[i] = sum / diag

    return 0

def computeSPMV(nx: int, nz: int, ny: int,
                A: torch.sparse.Tensor, x: torch.Tensor, y: torch.Tensor)-> int:
    
    n_rows = nx * ny * nz
    indices = A.indices()
    values = A.values()

    for i in range(n_rows):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]

        y[i] = torch.dot(nnz_values, x[nnz_cols])

    return 0

def computeRestriction(Afx: torch.Tensor, rf: torch.Tensor,
                       nc: int, f2c: torch.Tensor, rc: torch.Tensor)-> int:
    
    rc[:] = rf[f2c[:nc]] - Afx[f2c[:nc]]
    
    return 0

def computeProlongation(xf: torch.Tensor, xc: torch.Tensor, f2c: torch.Tensor, nc: int)-> int:
    
    xf[f2c[:]] += xc[:nc]
    
    return 0

def computeMG(nx: int, nz: int, ny: int,
              A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor,
              depth: int)-> int:


    # so there is a lot of stuff going on with mgData in the original. But it's already preprocessed there
    # so we can just assume that it's already done here and passed as this random thing called AmgData

    # in contrast to the reference implementation we compute the coarsening inside this function

    x = torch.zeros_like(x)
    ierr = 0

    if depth < 3:
        print("In the if clause, depth: ", depth)

        # first generate new matrix
        # since that might throw an error and we don't want to do computation in that case

        print("Starting to generate coarse problem, nx: ", nx, " ny: ", ny, " nz: ", nz)
        f2c_op, Ac, _ = generations.generate_coarse_problem(nx, ny, nz)
        print("End of generating coarse problem")

        nxc = nx // 2
        nyc = ny // 2
        nzc = nz // 2

        nc = nxc * nyc * nzc
        Axf = torch.zeros(nx * ny * nz)
        rc = torch.zeros(nc)
        xc = torch.zeros(nc)

        for i in range(num_pre_smoother_steps):
            ierr += computeSYMGS(nx, ny, nz, A, r, x)
        if ierr != 0:
            return ierr
        
        # Axf is where it is stored
        ierr = computeSPMV(nx, ny, nz, A, x, Axf)
        if ierr != 0:
            return ierr

        # Perform restriction operation using simple injection
        ierr = computeRestriction(Axf, r, nc, f2c_op, rc)
        if ierr != 0:
            return ierr
        
        # check that these attributes are correct

        ierr = computeMG(nxc, nyc, nzc, Ac, rc, xc, depth + 1)
        if ierr != 0:
            return ierr
        ierr = computeProlongation(x, xc, f2c_op, nc)
        if ierr != 0:
            return ierr
        
        for i in range(num_post_smoother_steps):
            ierr += computeSYMGS(nx, ny, nz, A, r, x)
        if ierr != 0:
            return ierr
        
    else:
        ierr = computeSYMGS(nx, ny, nz, A, r, x)
        if (ierr!=0):
            return ierr;  
    
    return 0



num = 32

A,y = generations.generate_torch_coo_problem(num,num,num)

x = torch.zeros(num*num*num)

computeMG(num, num, num, A,y,x,0)

