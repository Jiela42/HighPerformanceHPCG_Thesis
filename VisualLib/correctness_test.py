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



if __name__ == "__main__":

    check_coordinates(3, 3, 3)
    check_coordinates(4, 4, 4)
    check_coordinates(5, 5, 5)
    check_coordinates(5, 6, 7)
    check_coordinates(6, 6, 6)
    check_coordinates(8, 8, 8)
    check_coordinates(8, 16, 32)
    check_coordinates(16, 16, 16)
    print("All tests passed!")

    

