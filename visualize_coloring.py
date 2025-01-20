import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

# parameters we want to get analyzed for the stats
analyze_faces = False
num_col_row_stats = True


# first we read the colors
base_path = "colorings/"

def get_xyzc(file):
        
    xyzc = []
    with open(base_path + file, 'r') as f:
        dims = file.replace('.csv', '').replace('coloring_', '').replace('amgx_', '')
        nx = int(dims.split('x')[0])
        ny = int(dims.split('x')[1])
        nz = int(dims.split('x')[2])
        lines = f.readlines()
        colors = lines[0].split(',')
        colors = [color for color in colors if color != '' and color != '\n']
        for i, color in enumerate(colors):
            colors[i] = int(color)

            ix = i % nx
            iy = (i//nx) % ny
            iz = i // (nx * ny)

            xyzc.append((ix, iy, iz, int(color)))

    return xyzc

def get_dims(file):
    dims = file.replace('.csv', '').replace('coloring_', ''). replace('amgx_', '')
    nx = int(dims.split('x')[0])
    ny = int(dims.split('x')[1])
    nz = int(dims.split('x')[2])
    return nx, ny, nz

def rows_per_color_closed_form(c, nx, ny, nz):

    upper_bound_y = min(c//2, nx)
    sum_limit = min(upper_bound_y, 2 * (nz-1), 2 * (ny-1))

    # num_rows = sum_{y = 0}^{upper_bound_y} y//2
    # this formula can be computed
    num_rows = (((sum_limit-1)//2)**2 + ((sum_limit-1) //2) + ((sum_limit/2)**2) + (sum_limit//2))// 2

    # now we need to adjust for all the cases where we y//2 is at most nz-1 (because it is the max value)
    num_rows += (upper_bound_y - sum_limit) * min((nz - 1), (ny - 1))
    
    return num_rows
    
def rows_per_color_looping(c, nx, ny, nz):


    # if c == 0:
    #     return 1

    problem_color = -1


    
    upper_bound_ix = min(c+1, nx)
    c_mod2 = c % 2
    if c == problem_color:
        print("upper_bound_ix: ", upper_bound_ix, flush=True)


    cnt = 0
    for ix in range(c_mod2, upper_bound_ix, 2):
        upper_bound_iy = min((c-ix) // 2 + 1, ny)
        izctr = 0
        

        # we need to adjust y_start such that we don't expect nz to be bigger than possible
        iy_start = max ((c-ix) // 2 - 2 * (nz -1),0)

        if c == problem_color:
            print("c-ix//2) ", (c-ix )// 2, flush=True)
            print("2 * (nx -1) ", 2 * (nx -1), flush=True)
            
        # assert iy_start % 2 == 0

        # now we need to do the modulo adjustment (if c-ix) % 4 == 0 iy_start needs to be even otherwise odd
        # if (c-ix) % 4 == 0:
        #     if iy_start % 2 == 1:
        #         iy_start += 1
        #         # print("y_start has changed to even")
        #     # iy_start = iy_start if iy_start % 2 == 0 else iy_start + 1
        # else:
        #     if iy_start % 2 == 0:
        #         iy_start += 1
        #         # print("y_start has changed to odd")
        #     # iy_start = iy_start if iy_start % 2 == 1 else iy_start + 1

        # if iy_start % 2 == 0:
        #     print("iy_start is even", flush=True)

        if (iy_start % 2 == 1 and (c-ix) % 4 == 0) or (iy_start % 2 == 0 and (c-ix) % 4 != 0):
            iy_start += 1
# 
        

        if c == problem_color:
            print("ix-c_mod2%4: ", (ix - c_mod2) % 4, flush=True)
            print("ix: ", ix, flush=True)
            print("upper_bound_iy: ", upper_bound_iy, flush=True)
            print("y_start: ", iy_start, flush=True)
        for iy in range(iy_start,upper_bound_iy,2):
            if c==problem_color:
                print("iy: ", iy, flush=True)
            izctr += 1

        cnt += izctr
        # if c==4:
        #     print(f"rows_y: {izctr}, cnt: {cnt}",flush=True)

    return cnt

def has_duplicates(lst):
    return len(lst) != len(set(lst))

def get_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates

def check_color_computation_theory(x,y,z,colors, dims):


    # the theory is that color[i] = x + 2y + 4z
    print("Checking color computation theory for dims: ", dims)
    for i in range(len(x)):
        assert colors[i] == x[i] + 2*y[i] + 4*z[i]
        # print("Color: ", colors[i], " x: ", x[i], " y: ", y[i], " z: ", z[i])
        cxyz.append((colors[i], x[i], y[i], z[i]))
    
    
    # print("All colors are correct for dims: ", dims)

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

    # print("x: ", x)
    # print("y: ", y)
    # print("z: ", z)
    # print("colors: ", colors)

    # print("type x: ", type(x))
    # print("type y: ", type(y))
    # print("type z: ", type(z))
    # print("type colors: ", type(colors))

    output_file = "plots/coloring_plots/dims_" + dims + ".html"

    # check if the file already exists
    if os.path.exists(output_file):
        print("Skipping file: ", output_file)
        return

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
    # output_file = "plots/coloring_plots/dims_" + dims + ".html"
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def visualize_coloring(file):

    # print("Visualizing coloring for file: ", file)

    x = []
    y = []
    z = []

    cxyz = []

    # read the csv file in that order
    with open(base_path + file, 'r') as f:
        # print("Reading file: ", file)
        dims = file.replace('.csv', '').replace('coloring_', '').replace('amgx_', '')
        # print("Dims: ", dims)
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

            cxyz.append((int(color), ix, iy, iz, i))
    
    cxyz = sorted(cxyz, key=lambda x: x[0])

    print("Coloring for dims: ", dims)
    # for c, x, y, z, row in cxyz:
    #     if c == 4:
    #         print(f"Color: {c}, x: {x}, y: {y}, z: {z}, row: {row}")

    # check_xyz_coordinates(x, y, z, nx, ny, nz)
    # check_color_computation_theory(x, y, z, colors, dims)

    # now we print each x,y,z with the corresponding color
    # create_animation_buggy_old_version(x, y, z, colors, dims)
    name_adjustment = "_amgx" if "amgx" in file else ""
    print("Creating animation for dims: ", dims)
    create_animation(x, y, z, colors, dims + name_adjustment)

def get_color_stats(file):

    cxyz = []

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

            ix = i % nx
            iy = (i//nx) % ny
            iz = i // (nx * ny)

            cxyz.append((int(color), ix, iy, iz, i))
    
    if num_col_row_stats:
        max_color = max(colors)
        num_rows = len(colors)

        num_rows_per_color = [0] * (max_color + 1)

        for color in colors:
            num_rows_per_color[color] += 1

        max_num_rows_per_color = max(num_rows_per_color)
        last_row_color = colors[-1]
        min_num_rows_per_color = min(num_rows_per_color)
        num_colors_with_min_rows = num_rows_per_color.count(min_num_rows_per_color)
        num_colors_with_max_rows = num_rows_per_color.count(max_num_rows_per_color)
        colors_with_max_rows = [i for i, num_rows in enumerate(num_rows_per_color) if num_rows == max_num_rows_per_color]
        colors_with_min_rows = [i for i, num_rows in enumerate(num_rows_per_color) if num_rows == min_num_rows_per_color]


        temp_nx = nx if nx % 2 == 0 else nx - 1
        temp_ny = ny if ny % 2 == 0 else ny - 1
        temp_nz = nz if nz % 2 == 0 else nz - 1

        # print(f"option 1 {(nx * ny)//4 + (nx%2) + (ny%2)}")
        # print(f"option 2 {(nx * nz)//2 + (nx%2) + (nz%2)}")
        # print(f"option 3 {(ny * nz)}")

        calced_max_color = (nx - 1) + 2 * (ny-1) + 4 * (nz-1)
        
        num_rows_per_color_calced = []
        print("dims: ", dims)
        print(len(num_rows_per_color))
        print(calced_max_color+1)

        assert len(num_rows_per_color) == calced_max_color + 1

        for i in range(len(num_rows_per_color)):
            # if not (rows_per_color_closed_form(i, nx, ny, nz) == rows_per_color_looping(i, nx, ny, nz)):
            #     print("Error in closed form computation", flush=True)
            #     print("Color: ", i, flush=True)
            #     print("counted: ", num_rows_per_color[i], flush=True)
            #     print("Closed form: ", rows_per_color_closed_form(i, nx, ny, nz), flush=True)
            #     print("Looping: ", rows_per_color_looping(i, nx, ny, nz), flush=True)
            looping = rows_per_color_looping(i, nx, ny, nz)
            counted = num_rows_per_color[i]
            if not (looping == counted):
                print("Error in looping computation", flush=True)
                print("Color: ", i, flush=True)
                print("counted: ", counted, flush=True)
                print("Closed form: ", rows_per_color_closed_form(i, nx, ny, nz), flush=True)
                print("Looping: ", looping, flush=True)

            # if(i == 6 and dims == "6x5x3"):
            #     rows_with_color_6 = [(color, x, y, z) for color, x, y, z, row in cxyz if color == 6]
            #     print (f"Color 6: {rows_with_color_6}")
            assert(looping == counted)
            
            num_rows_for_color_i = rows_per_color_closed_form(i, nx, ny, nz)
            num_rows_per_color_calced.append(num_rows_for_color_i)

        # assert(num_rows_per_color_calced == num_rows_per_color)

        max_num_rows_per_color_calculated =min(
            (nx * ny)//4 + (nx*ny%4),
            (nx * nz)//2 + (nx*nz%2),
            (ny * nz)
        )

        print("**************************************************************************************************************************")
        print("File: ", file)
        print("Max color: ", max_color)
        print("Number of rows: ", num_rows)
        print("Max number of rows per color: ", max_num_rows_per_color)
        print("Number of colors with max rows: ", num_colors_with_max_rows)
        print(f"colors with max rows: {colors_with_max_rows}")
        print("Min number of rows per color: ", min_num_rows_per_color)
        print("Number of colors with min rows: ", num_colors_with_min_rows)
        print(f"colors with min rows: {colors_with_min_rows}")
        print("Last row color: ", last_row_color)
        print(f"Estimated Time Consumpution of colored compared to sequential:  {max_color/num_rows}")

        if max_num_rows_per_color != max_num_rows_per_color_calculated:
            print("Max number of rows per color calculated: ", max_num_rows_per_color_calculated)
            print("Actual max number of rows per color: ", max_num_rows_per_color)

        print("**************************************************************************************************************************")


    if analyze_faces:

        x0_face_colors = []
        y0_face_colors = []
        z0_face_colors = []
        xMax_face_colors = []
        yMax_face_colors = []
        zMax_face_colors = []

        print("Coloring for dims: ", dims)

        # print(cxyz)

        for color,x,y,z,row in cxyz:
            
            if x == 0:
                x0_face_colors.append(color)
            if y == 0:
                y0_face_colors.append(color)
            if z == 0:
                z0_face_colors.append(color)
            if x == nx - 1:
                xMax_face_colors.append(color)
            if y == ny - 1:
                yMax_face_colors.append(color)
            if z == nz - 1:
                zMax_face_colors.append(color)

        if(has_duplicates(x0_face_colors)):
            print("x0 has duplicates: ", get_duplicates(x0_face_colors))
        
        if(has_duplicates(y0_face_colors)):
            print("y0 has duplicates: ", get_duplicates(y0_face_colors))
        
        if(has_duplicates(z0_face_colors)):
            print("z0 has duplicates: ", get_duplicates(z0_face_colors))
        
        if(has_duplicates(xMax_face_colors)):
            print("xMax has duplicates: ", get_duplicates(xMax_face_colors))
        
        if(has_duplicates(yMax_face_colors)):
            print("yMax has duplicates: ", get_duplicates(yMax_face_colors))
        
        if(has_duplicates(zMax_face_colors)):
            print("zMax has duplicates: ", get_duplicates(zMax_face_colors))
        
        # assert(not has_duplicates(x0_face_colors))

def from_raw_to_csv(raw_file):
    csv_file = raw_file.replace('raw_', '').replace('.txt', '.csv')
    with open(raw_file, 'r') as infile, open(csv_file, 'w') as outfile:
        colors = []
        for line in infile:
            color = line.split()[-1]  # Get the last element which is the color digit
            colors.append(color)
        outfile.write(','.join(colors))   

def check_distance(x, y, z, colors, nx, ny, nz):
    print("Checking distance between colors for dims: ", nx, ny, nz, flush=True)
    xyzc = list(zip(x, y, z, colors))

    for ix, iy, iz, color in xyzc:
        # print(f"Checking color: {color} at {ix, iy, iz}", flush=True)
        # check neighbours if they have the same color
        neighbors = []
        for sz in range(-1, 2):
            for sy in range(-1, 2):
                for sx in range(-1, 2):
                    if sz + sy + sx != 0 and ix+sx > -1 and ix+sx < nx and iy+sy > -1 and iy+sy < ny and iz+sz > -1 and iz+sz < nz:
                        neighbors.append((ix+sx, iy+sy, iz+sz))

    
        for nxi, nyi, nzi in neighbors:
            if (nxi, nyi, nzi, color) in xyzc:
                print(f"Found neighbor with same color: {color} at {nxi, nyi, nzi}")
        
# grab all the files in the directory
files = os.listdir(base_path)

csv_files = [file for file in files if file.endswith('.csv')]

# first check if there is raw data that needs to be converted to csv
raw_files = [file for file in files if file.endswith('.txt')]
for raw_file in raw_files:
    new_name = raw_file.replace('raw_', '').replace('.txt', '.csv')
    if new_name not in csv_files:
        print(f"Converting raw file: {raw_file} to csv")
        from_raw_to_csv(base_path + raw_file)

amgx_files = [file for file in csv_files if 'amgx' in file]

# visualize_coloring(files[0])

# toy_example()
files_to_ignore = [
    # "coloring_4x4x4.csv",
    "coloring_8x8x8.csv",
    "coloring_16x16x16.csv",
    "coloring_24x24x24.csv",
    "coloring_32x32x32.csv",
    "coloring_64x64x64.csv"
]

files_to_color= [
    # "coloring_32x32x32.csv",
    # "coloring_64x64x64.csv"
]

files_to_inspect=[
    "coloring_3x4x5.csv",
    "coloring_4x3x5.csv",
    "coloring_5x4x3.csv",
    "coloring_5x3x4.csv",
    "coloring_3x5x4.csv",
    "coloring_4x5x3.csv",
    "coloring_4x5x6.csv",
    "coloring_5x4x6.csv",
    "coloring_5x6x4.csv",
    "coloring_4x6x5.csv",
    "coloring_6x5x4.csv",
    "coloring_6x4x5.csv",
    "coloring_6x5x6.csv",
    "colorings_5x3x6.csv",
    "coloring_5x6x3.csv",
    "coloring_3x5x6.csv",
    "coloring_6x3x5.csv",
    "coloring_6x5x3.csv",
    "coloring_3x6x5.csv",
    
]

for file in amgx_files:
    # visualize_coloring(file)
    xyzc = get_xyzc(file)
    
    # Unzipping xyzc into independent lists
    x, y, z, colors = zip(*xyzc)
    
    # Convert tuples to lists if needed
    x = list(x)
    y = list(y)
    z = list(z)
    colors = list(colors)

    nx, ny, nz = get_dims(file)
    if (nx, ny, nz) == (16, 16, 16):
    
        check_distance(x, y, z, colors, nx, ny, nz)


# for file in files_to_color:
#     visualize_coloring(file)

# for file in files_to_color:
#     visualize_coloring(file)

# visualize_coloring("coloring_4x4x4.csv")

# for file in files:
#     if file not in files_to_ignore:
        # visualize_coloring(file)
        # get_color_stats(file)
    # visualize_coloring(file)
    # get_color_stats(file)

# for file in files_to_inspect:
#     if file in files:
#         get_color_stats(file)

# get_color_stats("coloring_4x3x5.csv")
# get_color_stats("coloring_6x5x3.csv")

# get_color_stats("coloring_8x8x8.csv")
