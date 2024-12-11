import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# first we read the colors
base_path = "colorings/"


def visualize_coloring(file):

    x = []
    y = []
    z = []

    # read the csv file in that order
    with open(base_path + file, 'r') as f:
        dims = file.replace('.csv', '').replace('coloring_', '')
        nx = int(dims.split('x')[0])
        ny = int(dims.split('x')[1])
        nz = int(dims.split('x')[2])
        lines = f.readlines()
        colors = lines[0].split(',')
        colors = [color for color in colors if color != '' and color != '\n']
        for i, color in enumerate(colors):
            colors[i] = int(color)

            # now we need to turn i into x, y, z
            # i = ix + nx * iy + nx * ny * iz
            ix = i % nx
            iy = (i//nx) % ny
            iz = i // (nx * ny)

            x.append(ix)
            y.append(iy)
            z.append(iz)
    
    # now we print each x,y,z with the corresponding color

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis')

    # Add color bar
    color_bar = plt.colorbar(scatter, ax=ax)
    color_bar.set_label('Color Value')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    output_file = "plots/coloring_plots/dims_" + dims + ".png"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)     



# grab all the files in the directory
files = os.listdir(base_path)

visualize_coloring(files[0])

# for file in files:
#     visualize_coloring(file)

