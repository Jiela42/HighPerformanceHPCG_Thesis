import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go



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

    # Create a 3D scatter plot using plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title='Color Value')
            )
        )])

        # Set labels
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title=f'3D Coloring Visualization for {dims}'
        )

        # Save plot to HTML file
        output_file = "plots/coloring_plots/dims_" + dims + ".html"
        fig.write_html(output_file)



# grab all the files in the directory
files = os.listdir(base_path)

visualize_coloring(files[0])

# for file in files:
#     visualize_coloring(file)

