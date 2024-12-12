import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np



# first we read the colors
base_path = "colorings/"

def check_xyz_coordinates(x, y, z, nx, ny, nz):

    i_xyz = {}

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = ix + nx*iy + nx*ny*iz
                i_xyz[i] = (ix, iy, iz)

    for i in range(len(x)):
        ix, iy, iz = i_xyz[i]
        assert x[i] == ix
        assert y[i] == iy
        assert z[i] == iz
    
    print("All x, y, z coordinates are correct")


def create_animation_buggy_old_version(x, y, z, colors, dims):


    # Create frames for the animation
    frames = []
    max_color = max(colors)
    min_color = min(colors)

    min_x = min(x) - 1
    max_x = max(x) + 1
    min_y = min(y) - 1
    max_y = max(y) + 1
    min_z = min(z) - 1
    max_z = max(z) + 1


    for color in range(1, max_color + 1):
        frame_colors = []
        frame_x = []
        frame_y = []
        frame_z = []
        for i in range(len(x)):
            if colors[i] <= color:
                frame_x.append(x[i])
                frame_y.append(y[i])
                frame_z.append(z[i])
                frame_colors.append(colors[i])
        
        frames.append(go.Frame(data=[go.Scatter3d(
            x=frame_x,
            y=frame_y,
            z=frame_z,
            mode='markers',
            marker=dict(
                size=5,
                color=frame_colors,
                colorscale='Turbo',
                colorbar=dict(title='Color Value'),
                cmin=min_color,
                cmax=min_color
            )
        )]))

    # Create the initial plot
    fig = go.Figure(
        data=[go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='markers',
            marker=dict(
                size=5,
                color=[],
                colorscale='Turbo',
                colorbar=dict(title='Color Value'),
                cmin=min_color,
                cmax=max_color
            )
        )],
        layout=go.Layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[min_x, max_x]),
                yaxis=dict(range=[min_y, max_y]),
                zaxis=dict(range=[min_z, max_z]),
                camera=dict(  # Add this line to fix the camera view
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.25, y=1.25, z=1.25)
                )

            ),
            title=f'3D Coloring Visualization for {dims}',
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    },
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
        ),
        frames=frames
    )


    # Save plot to HTML file
    output_file = "plots/coloring_plots/dims_" + dims + ".html"
    fig.write_html(output_file)



def create_animation(x, y, z, colors, dims):

    min_x = min(x) - 1
    max_x = max(x) + 1
    min_y = min(y) - 1
    max_y = max(y) + 1
    min_z = min(z) - 1
    max_z = max(z) + 1

    max_color = max(colors)
    min_color = min(colors)

    print("max_color: ", max_color)

    frames = []

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    )

    for c in range(max_color + 1):
        print("Creating frame for color: ", c)
        mask = colors == c
        frame_x = []
        frame_y = []
        frame_z = []
        frame_colors = []
        frame_text = []
        for i in range(len(x)):
            if colors[i] <= c:
                frame_x.append(x[i])
                frame_y.append(y[i])
                frame_z.append(z[i])
                frame_colors.append(colors[i])
                frame_text.append(str(colors[i]))
        frame = go.Frame(data=[go.Scatter3d(
            x=frame_x,
            y=frame_y,
            z=frame_z,
            mode='markers+text',
            text=frame_text,
            textposition='middle center',
            textfont=dict(size=20,color='white', family='Arial Black'),
            marker=dict(
                size=6,
                color=frame_colors,
                colorscale='Turbo',
                cmin=min_color,
                cmax=max_color
            )
        )],
        layout=dict(scene=dict(camera=camera,
                               xaxis=dict(range=[min_x, max_x], autorange=False),
                yaxis=dict(range=[min_y, max_y], autorange=False),
                zaxis=dict(range=[min_z, max_z], autorange=False),
                aspectmode='cube'))
        )
        frames.append(frame)

    # Create the figure
    fig = go.Figure(
        data=[go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='markers+text',
            textposition='middle center',
            textfont=dict(size=20, color='white', family='Arial Black'),
            text=[],
            marker=dict(
                size=6,
                color=[],
                colorscale='Turbo',
                cmin=min_color,
                cmax=max_color
            )
        )],
        layout=go.Layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[min_x, max_x], autorange=False),
                yaxis=dict(range=[min_y, max_y], autorange=False),
                zaxis=dict(range=[min_z, max_z], autorange=False),
                aspectmode='cube',
                camera=dict(  # Add this line to fix the camera view
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.25, y=1.25, z=1.25)
                )
            ),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        ),
        frames=frames
    )

    # Save plot to HTML file
    output_file = "plots/coloring_plots/dims_" + dims + ".html"
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)



def visualize_coloring(file):

    print("Visualizing coloring for file: ", file)

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
    
    # check_xyz_coordinates(x, y, z, nx, ny, nz)

    # now we print each x,y,z with the corresponding color
    # create_animation_buggy_old_version(x, y, z, colors, dims)
    create_animation(x, y, z, colors, dims)

def get_color_stats(file):
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
    
    max_color = max(colors)
    num_rows = len(colors)

    num_rows_per_color = [0] * (max_color + 1)

    for color in colors:
        num_rows_per_color[color] += 1

    max_num_rows_per_color = max(num_rows_per_color)

    print("**************************************************************************************************************************")
    print("File: ", file)
    print("Max color: ", max_color)
    print("Number of rows: ", num_rows)
    print("Max number of rows per color: ", max_num_rows_per_color)
    print(f"Estimated Time Consumpution of colored compared to sequential:  {max_color/num_rows}")
    print("**************************************************************************************************************************")

        
# grab all the files in the directory
files = os.listdir(base_path)

# visualize_coloring(files[0])

# toy_example()
files_to_ignore = [
    # "coloring_32x32x32.csv",
    "coloring_64x64x64.csv"
]

files_to_color= [
    # "coloring_32x32x32.csv",
    # "coloring_64x64x64.csv"
]

for file in files_to_color:
    visualize_coloring(file)


for file in files:
    if file not in files_to_ignore:
        visualize_coloring(file)
    get_color_stats(file)

