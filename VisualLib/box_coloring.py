import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HighPerformanceHPCG_Thesis.VisualLib.visualize_coloring import create_animation


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




def main():

    bx = 4
    by = 3
    bz = 7

    # generate_box_coloring(4, 4, 4, bx, by, bz)

    analyze_box_coloring(3, 3, 3, bx, by, bz)
    analyze_box_coloring(4, 4, 4, bx, by, bz)
    analyze_box_coloring(4,5,6, bx, by, bz)
    analyze_box_coloring(6,5,4, bx, by, bz)
    analyze_box_coloring(5,4,6, bx, by, bz)
    analyze_box_coloring(8, 8, 8, bx, by, bz)
    analyze_box_coloring(16, 16, 16, bx, by, bz)


if __name__ == "__main__":
    main()