# for (int i = 0; i < num_elements; i++) {
#     int ix = i % nx;
#     int iy = (i / nx) % ny;
#     int iz = i / (nx * ny);


def check_coordinates(nx, ny, nz):
    
    original_loop = []
    new_loop = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = iz * nx * ny + iy * nx + ix
                original_loop.append((ix, iy, iz, i))
    
    for i in range(nx * ny * nz):
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)
        new_loop.append((ix, iy, iz, i))

    # turn loops into sets
    original_loop = set(original_loop)
    new_loop = set(new_loop)

    # check if the loops are the same
    assert original_loop == new_loop, f"Loops are not the same for nx={nx}, ny={ny}, nz={nz}"

def check_f2c_op(nx, ny, nz):

    assert nx % 2 == 0 and ny % 2 == 0 and nz % 2 == 0, "nx, ny, nz must be even"


    nxf = nx
    nyf = ny
    nzf = nz

    nxc = nx // 2
    nyc = ny // 2
    nzc = nz // 2

    num_fine_rows = nxf * nyf * nzf
    num_coarse_rows = nxc * nyc * nzc

    hpcg_original_loop = [-1 for _ in range(num_fine_rows)]
    single_loop = [-1 for _ in range(num_fine_rows)]

    # compute the single loop version
    for coarse_idx in range(num_coarse_rows):
        izc = coarse_idx // (nxc * nyc)
        iyc = (coarse_idx % (nxc * nyc)) // nxc
        ixc = coarse_idx % nxc

        izf = izc * 2
        iyf = iyc * 2
        ixf = ixc * 2

        fine_idx = ixf + nxf * iyf + nxf * nyf * izf
        single_loop[coarse_idx] = fine_idx

    
    # compute the HPCG version

    for izc in range(nzc):
        for iyc in range(nyc):
            for ixc in range(nxc):
                izf = 2 * izc
                iyf = 2 * iyc
                ixf = 2 * ixc
                currentCoarseRow = izc * nxc * nyc + iyc * nxc + ixc
                currentFineRow = izf * nxf * nyf + iyf * nxf + ixf
                hpcg_original_loop[currentCoarseRow] = currentFineRow

    print("Single Loop: ", single_loop)
    print("HPCG Loop: ", hpcg_original_loop)

    # compare the two arrays
    assert single_loop == hpcg_original_loop, f"Loops are not the same for nx={nx}, ny={ny}, nz={nz}"


if __name__ == "__main__":

    # check_coordinates(3, 3, 3)
    # check_coordinates(4, 4, 4)
    # check_coordinates(5, 5, 5)
    # check_coordinates(5, 6, 7)
    # check_coordinates(6, 6, 6)
    # check_coordinates(8, 8, 8)
    # check_coordinates(8, 16, 32)
    # check_coordinates(16, 16, 16)

    # check_f2c_op(4, 4, 4)
    # check_f2c_op(6, 6, 6)
    # check_f2c_op(8, 8, 8)
    # check_f2c_op(8, 16, 32)
    # check_f2c_op(16, 16, 16)
    check_f2c_op(24, 24, 24)
    # check_f2c_op(32, 32, 32)
    # check_f2c_op(64, 64, 64)

    print("All tests passed!")

    

