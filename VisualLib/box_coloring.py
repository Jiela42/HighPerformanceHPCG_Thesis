import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HighPerformanceHPCG_Thesis.VisualLib.visualize_coloring import create_animation

def print_dense_version_of_striped_matrix(num_rows, j_min_i):
    num_stripes = len(j_min_i)
    matrix = [[0 for i in range(num_rows)] for j in range(num_rows)]
    for i in range(num_rows):
        for j in range(num_stripes):
            neighbor = i + j_min_i[j]
            if neighbor < num_rows and neighbor >= 0:
                matrix[i][neighbor] = 1
    for i in range(num_stripes):
        print(matrix[i])

def generate_box_coloring(nx, ny, nz, bx, by, bz):

    assert bx > 2 and by > 2 and bz > 2, "Box size must be greater than 2, otherwise dependencies are violated"

    x = []
    y = []
    z = []
    colors = []

    print(f"Generating box coloring for {nx}x{ny}x{nz} with box size {bx}x{by}x{bz}")

    for inx in range(nx):
        for iny in range(ny):
            for inz in range(nz):
                mod_inx = inx % bx
                mod_iny = iny % by
                mod_inz = inz % bz

                x.append(inx)
                y.append(iny)
                z.append(inz)
                
                # faces are in the z direction
                # rows are in the y direction
                # cols are in the x direction

                color = mod_inx + mod_iny * bx + mod_inz * bx * by

                colors.append(color)
    
    dims = f"{nx}x{ny}x{nz}" + "_box_coloring"
    create_animation(x, y, z, colors, dims)

    return x, y, z, colors

def analyze_box_coloring(nx, ny, nz, bx, by, bz):
    
    x,y,z,color = generate_box_coloring(nx, ny, nz, bx, by, bz)

    xyzc = list(zip(x, y, z, color))

    print(f"Dimensions: {nx}x{ny}x{nz}")
    print(f"Box size: {bx}x{by}x{bz}")

    calculated_num_colors = min(bx,nx) * min(by,ny) * min(bz,nz)
    
    assert len(set(color)) == calculated_num_colors, f"Number of colors is not correct. Expected: {len(set(color))}, Got: {calculated_num_colors}"

    # Check if each color has the correct number of nodes
    for c in range(calculated_num_colors):
        num_cols = nx // bx
        num_rows = ny // by
        num_faces = nz // bz

        offs_x = c % bx
        offs_y = (c-offs_x) % (bx * by) // bx
        offs_z = (c - offs_x - offs_y * bx) // (bx * by)

        # faces are in the z direction
        # rows are in the y direction
        # cols are in the x direction

        # print(f"Color {c} has offset ({offs_x}, {offs_y}, {offs_z})")

        col_adjustment = 1 if offs_x < nx % bx else 0
        row_adjustment = 1 if offs_y < ny % by else 0
        face_adjustment = 1 if offs_z < nz % bz else 0

        # print(f"Face adjustment: {face_adjustment}, Col adjustment: {col_adjustment}, Row adjustment: {row_adjustment}")

        adjusted_faces = num_faces + face_adjustment
        adjusted_cols = num_cols + col_adjustment
        adjusted_rows = num_rows + row_adjustment

        # figure out how to adjust to make sure we don't expect out of dims nodes

        # print(f"Color {c} has {adjusted_faces} faces (z), {adjusted_cols} cols (x), {adjusted_rows} rows(y)")

        num_nodes_with_color = adjusted_faces * adjusted_rows * adjusted_cols
        assert color.count(c) == num_nodes_with_color, f"Color {c} has {num_nodes_with_color} nodes, expected {color.count(c)}"

        # Check if we can easily find all nodes with a given color
        for iy in range(adjusted_rows):
            for ix in range(adjusted_cols):
                for iz in range(adjusted_faces):
                    # color = mod_inx + mod_iny * bx + mod_inz * bx * by
                    calculated_x = ix * bx + offs_x
                    calculated_y = iy * by + offs_y
                    calculated_z = iz * bz + offs_z

                    if (calculated_x, calculated_y, calculated_z, c) not in xyzc:
                        print(f"Node ({calculated_x}, {calculated_y}, {calculated_z}) with color {c} not found")
                        assert False, "Node not found in tripple loop"

        for i in range(num_nodes_with_color):

            iy = i // (adjusted_cols * adjusted_faces)
            ix = (i // adjusted_faces) % adjusted_cols
            iz = i % adjusted_faces

            calculated_x = ix * bx + offs_x
            calculated_y = iy * by + offs_y
            calculated_z = iz * bz + offs_z

            if (calculated_x, calculated_y, calculated_z, c) not in xyzc:
                print(f"Node ({calculated_x}, {calculated_y}, {calculated_z}) with color {c} not found")
                assert False, "Node not found in single loop"

    # check that every node has exactly one color
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                node = (ix, iy, iz)
                colors = [c for x, y, z, c in xyzc if (x, y, z) == node]
                assert len(colors) == 1, f"Node {node} has multiple colors: {colors}"

    
    vector_matrix = [i for i in range(nx*ny*nz)]

    for i in range(nx*ny*nz):
        # get the color of the node
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)

        row = ix + iy * nx + iz * nx * ny

        assert row == i, f"Row {row} is not equal to i {i}"
        node_color = [c for x, y, z, c in xyzc if (x, y, z) == (i % nx, (i // nx) % ny, i // (nx * ny))][0]
        if node_color == 0:
            vector_matrix[i] = -42

    for i in range(ny * nz):
        row = vector_matrix[i * nx: (i+1) * nx]
        formatted_row = " ".join(f"{num:4}" for num in row)  # Adjust the width as needed
        print(formatted_row, flush=True)

        


def analyze_general_box_coloring(num_rows, j_min_i, colors):

    num_stripes = len(j_min_i)
    print(f"stripes: {j_min_i}")

    # confirm that no neighbor is colored the same
    for i in range(num_rows):
        for j in range(num_stripes):
            neighbor = i + j_min_i[j]
            if neighbor < num_rows and neighbor >= 0 and neighbor != i:
                assert colors[i] != colors[neighbor], f"Neighbor {neighbor} of node {i} has the same color {colors[i]}"

    print("analisys passed for general box coloring")

def general_striped_box_coloring(num_rows, j_min_i):

    print_dense_version_of_striped_matrix(num_rows, j_min_i)

    num_stripes = len(j_min_i)
    colors = [-1] * num_rows
    # adapting the starting rows ensures that row zero gets colored
    starting_rows = [x - j_min_i[0] for x in j_min_i]

    num_patterns = 1
    # this stores the pattern of the stripes, i.e. the length and the starting color
    patterns = [(1,0)]

    for i in range(1, num_stripes):
        # every time the new j_min_i is increased by more than just 1 we get a new pattern
        if j_min_i[i] - 1 != j_min_i[i-1]:
            num_patterns += 1
            patterns.append((1, i))
        else:
            patterns[-1] = (patterns[-1][0] + 1, patterns[-1][1])
    

    # first we set the initial colors
    for i in range(num_stripes):
        colors[starting_rows[i]] = i

    # how do we know what the end of the last pattern is?
    # in the box coloring for the stencil we repeat the pattern 3 times, because
    # its a 2D pattern and we have 3 dimensions,
    # we need to repeat 3x because that's how many rows we are depending on or how many rows it takes to fill a face of the box
    # how do we know how many times to repeat the pattern in the general case?
    # i can check my neighbour to see if it's the same color, if it is then i know i'm at the end of the pattern
    # I really don't like this iterative dependency on previous colors, but let's see

    print(f"colors after initial coloring: {colors}")

    # then we set the colors for the rest of the rows (row zero is already colored)
    for i in range(1, num_rows):
        # if the row isn't colored
        if colors[i] == -1:
            # find the pattern this row is in
            prev_row_color = colors[i-1]
            for pattern in range(len(patterns)):
                pattern_len, pattern_start = patterns[pattern]
                if prev_row_color >= pattern_start and prev_row_color < pattern_start + pattern_len:
                   
                    # we apply the next color in the pattern
                    color = prev_row_color + 1 if prev_row_color + 1 < pattern_start + pattern_len else pattern_start
            
                    if(color == 3):
                        print(f"i: {i}, prev_row_color: {prev_row_color}, pattern: {pattern}, pattern_len: {pattern_len}, pattern_start: {pattern_start}")
                        # print(f"colors: {colors}")

                    # check if one of the neighbors has the same color
                    for j in range(num_stripes):
                        neighbor = i + j_min_i[j]
                        if i== 3:
                            print(f"neighbor: {neighbor}, color: {color}, colors[neighbor]: {colors[neighbor]}")
                        if neighbor < num_rows and neighbor >= 0 and colors[neighbor] == color:
                            # if the neighbor has the same color, we know we are at the end of the pattern and we need to start a new one
                            if pattern < len(patterns) - 1:
                                next_pattern_len, next_pattern_start = patterns[pattern+1]
                                color = next_pattern_start
                            else:
                                # print("we are at the last pattern")
                                # print(f"pattern: {pattern}, patterns: {patterns}, pattern_len: {pattern_len}, pattern_start: {pattern_start}")
                                
                                # if we are at the last pattern, we need to start over
                                color = 0
                            break
                    colors[i] = color
                    break

    
    print(f"patterns: {patterns}")
    print(f"colors: {colors}")
    return colors             


def nils_offset_analysis(nx, ny, nz, bx, by, bz):

    row_color_xyz = []
        
    num_colors = bx * by * bz

    for color in range(num_colors):

        num_color_cols = nx // bx
        num_color_rows = ny // by
        num_color_faces = nz // bz

        color_offs_x = color % bx
        color_offs_y = (color - color_offs_x) % (bx * by) // bx
        color_offs_z = (color - color_offs_x - bx * color_offs_y) // (bx * by)


        num_color_cols = (num_color_cols + 1) if (color_offs_x < nx % bx) else num_color_cols
        num_color_rows = (num_color_rows + 1) if (color_offs_y < ny % by) else num_color_rows
        num_color_faces = (num_color_faces + 1) if (color_offs_z < nz % bz) else num_color_faces

        num_nodes_with_color = num_color_cols * num_color_rows * num_color_faces

        for i in range(num_nodes_with_color):

            # find out the position of the node (only considering faces, cols and rows that actually have that color)
            ix = i % num_color_cols
            iy = ((i % (num_color_cols * num_color_rows))) // num_color_cols
            iz = i // (num_color_cols * num_color_rows)
            
            # adjust the counter to the correct position when all nodes are considered
            ix = ix * bx + color_offs_x
            iy = iy * by + color_offs_y
            iz = iz * bz + color_offs_z

            row = ix + iy * nx + iz * nx * ny

            row_color_xyz.append((row, color, ix, iy, iz))
    
    # check if they respect direct dependencies
    for i in range(len(row_color_xyz)):
        row, color, ix, iy, iz = row_color_xyz[i]
        
        # make a list of neighbors
        neighbors = []

        for jx in range(-1, 2):
            for jy in range(-1, 2):
                for jz in range(-1, 2):
                    if ix + jx >= 0 and ix + jx < nx and iy + jy >= 0 and iy + jy < ny and iz + jz >= 0 and iz + jz < nz:
                        neighbor_row = (ix + jx) + (iy + jy) * nx + (iz + jz) * nx * ny
                        neighbors.append((neighbor_row, color, ix + jx, iy + jy, iz + jz))
        
        # check that no neighbor is in the collection of color
        # because we added the first persons color that means all neighbors have different colors
        
        assert(neighbors not in row_color_xyz), f"Neighbor of node {row} has the same color {color}"
    
    for i in range(nx * ny * nz):
        # check if there are rows with multiple colors
        row = [x for x in row_color_xyz if x[0] == i]
        assert len(row) <= 1, f"Row {i} has multiple colors: {row}"

    # check that every row is colored twice
    for i in range(nx * ny * nz):
        # find row i
        row = [x for x in row_color_xyz if x[0] == i]
        assert len(row) >= 1, f"Row {i} is not colored: {row}"
    

    




    # visualize the nils color
    x = []
    y = []
    z = []
    colors = []

    for row, color, ix, iy, iz in row_color_xyz:
        x.append(ix)
        y.append(iy)
        z.append(iz)
        colors.append(color)
    
    dims = f"{nx}x{ny}x{nz}" + "_nils_offset_analysis"
    # create_animation(x, y, z, colors, dims)

def general_striped_box_coloring_for_3D27pt(nx):

    # first we generate the stripes
    num_stripes = 27
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

    for i in range(num_stripes):
        off_x, off_y, off_z = neighbour_offsets[i]
        j_min_i.append(off_x + nx*off_y + nx*nx*off_z)

    colors = general_striped_box_coloring(nx*nx*nx, j_min_i)
    resorted_colors = []

    ny = nx
    nz = nx

    x = []
    y = []
    z = []

    for inx in range(nx):
        for iny in range(ny):
            for inz in range(nz):

                x.append(inx)
                y.append(iny)
                z.append(inz)

                i = inx + nx*iny + nx*nx*inz
                resorted_colors.append(colors[i])
    
    dims = f"{nx}x{ny}x{nz}" + "_general_striped_box_coloring_for_3D27pt"
    create_animation(x, y, z, resorted_colors, dims)
    analyze_general_box_coloring(nx*nx*nx, j_min_i, colors)
    


def main():

    bx = 3
    by = 3
    bz = 3

    # generate_box_coloring(4, 4, 4, bx, by, bz)

    # analyze_box_coloring(3, 3, 3, bx, by, bz)
    # analyze_box_coloring(4, 4, 4, bx, by, bz)
    # analyze_box_coloring(4,5,6, bx, by, bz)
    # analyze_box_coloring(6,5,4, bx, by, bz)
    # analyze_box_coloring(5,4,6, bx, by, bz)
    # analyze_box_coloring(8, 8, 8, bx, by, bz)
    # analyze_box_coloring(12, 12, 6, bx, by, bz)
    # analyze_box_coloring(16, 16, 16, bx, by, bz)

    # general_striped_box_coloring_for_3D27pt(4)

    nils_offset_analysis(3, 3, 3, 3, 3, 3)
    nils_offset_analysis(4, 4, 4, 3, 3, 3)
    nils_offset_analysis(8, 8, 8, 3, 3, 3)
    nils_offset_analysis(9,9,9, 3, 3, 3)
    nils_offset_analysis(16, 16, 16, 3, 3, 3)
    # nils_offset_analysis(32, 32, 32, 3,3,3)

    print("All tests passed")


if __name__ == "__main__":
    main()