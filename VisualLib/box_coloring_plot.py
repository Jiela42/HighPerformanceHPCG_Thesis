import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import math

save_path = "plots/paper_figures/"
color_scale = "turbo"

def generate_2D_box_coloring(nx, ny, bx, by):

    assert bx >= 2 and by >= 2, "Box size must be greater than 2, otherwise dependencies are violated"

    x = []
    y = []
    colors = []

    print(f"Generating box coloring for {nx}x{ny} with box size {bx}x{by}")

    for inx in range(nx):
        for iny in range(ny):
                mod_inx = inx % bx
                mod_iny = iny % by

                x.append(inx)
                y.append(iny)
                
                # faces are in the z direction
                # rows are in the y direction
                # cols are in the x direction

                color = mod_inx + mod_iny * bx

                colors.append(color)

    return x, y, colors

def generate_2D_propagated_coloring(nx, ny):
     
    x = []
    y = []
    colors = []

    print(f"Generating propagated coloring for {nx}x{ny}")

    for inx in range(nx):
        for iny in range(ny):
                color = inx + 2*iny
                x.append(inx)
                y.append(iny)
                colors.append(color)

    return x, y, colors

def generate_3D_propagated_coloring(nx, ny, nz):
    x = []
    y = []
    z = []
    colors = []

    print(f"Generating propagated coloring for {nx}x{ny}x{nz}")

    for inx in range(nx):
        for iny in range(ny):
            for inz in range(nz):
                color = inx + 2*iny + 4*inz
                x.append(inx)
                y.append(iny)
                z.append(inz)
                colors.append(color)

    return x, y, z, colors

def generate_3D_box_coloring(nx, ny, nz, bx, by, bz):
    assert bx >= 2 and by >= 2 and bz >= 2, "Box size must be greater than 2, otherwise dependencies are violated"

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

                color = mod_inx + mod_iny * bx + mod_inz * bx * by

                colors.append(color)

    return x, y, z, colors

def plot_2D_box_coloring(nx, ny, bx, by):

    x, y, colors = generate_2D_box_coloring(nx, ny, bx, by)

    # Add pale grey gridlines
    plt.grid(color='lightgrey', linestyle='--', linewidth=1.0, zorder=0.0)

    rect_color = "#bbacbd"

    # add boxes around each bx*by region
    box_buffer = 0.3
    ax = plt.gca()
    for box_x in range(0, nx, bx):
        for box_y in range(0, ny, by):
            
            rect = patches.FancyBboxPatch(
                 (box_x - box_buffer, box_y - box_buffer),
                 bx-1 + 2 * box_buffer, by-1 + 2 * box_buffer,
                 boxstyle="round,pad=0.1, rounding_size=0.3",
                 linewidth=1, edgecolor= rect_color, facecolor=rect_color, zorder=-5.0)
            ax.add_patch(rect)


    # Set x and y limits
    plt.xlim(-0.5, nx-0.5)
    plt.ylim(-0.5, ny-0.5)

    # Set ticks
    plt.xticks(np.arange(0, nx, 1))
    plt.yticks(np.arange(0, ny, 1))

    # Remove the enclosing box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    # Create the scatter plot
    plt.scatter(x, y, c=colors, cmap=color_scale, s=800, edgecolor='black', zorder=5.0)

    # Add color values as text on the dots
    for i in range(len(x)):
        plt.text(x[i], y[i], str(colors[i]), color='white', fontsize=15, ha='center', va='center', zorder = 10)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    # plt.title(f'Box Coloring (ColoringBox {bx}x{by})')

    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    save_path_figure = save_path + f"2D_box_coloring_{bx}x{by}.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()

def plot_2D_propagated_coloring(nx, ny):
    
    x,y, colors = generate_2D_propagated_coloring(nx, ny)

    # Add pale grey gridlines
    plt.grid(color='lightgrey', linestyle='--', linewidth=1.0, zorder=0.0)

    # Set x and y limits
    plt.xlim(-0.5, nx-0.5)
    plt.ylim(-0.5, ny-0.5)

    # Set ticks
    plt.xticks(np.arange(0, nx, 1))
    plt.yticks(np.arange(0, ny, 1))

    # Remove the enclosing box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    # Create the scatter plot
    plt.scatter(x, y, c=colors, cmap=color_scale, s=800, edgecolor='black', zorder=5.0)

    # Add color values as text on the dots
    for i in range(len(x)):
        plt.text(x[i], y[i], str(colors[i]), color='white', fontsize=15, ha='center', va='center', zorder = 10)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    # plt.title(f'Box Coloring (ColoringBox {bx}x{by})')

    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    save_path_figure = save_path + f"2D_propagated_coloring.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()

def plot_2D_propagated_coloring_COR_format(nx, ny):
    x, y, colors = generate_2D_propagated_coloring(nx, ny)

    # make a dictionary of colors
    color_dict = {}
    for i in range(len(x)):
        color_dict[(x[i], y[i])] = colors[i]

    # this stores the color of each cell/row
    ic = [0 for i in range(nx*ny)]

    for i in range(nx*ny):
        # if i = ix+iy*nx
        ix = i % nx
        iy = (i-ix) // nx

        ic[i] = color_dict[(ix, iy)]


    # create COR format

    max_color = max(ic)
    num_colors = max_color + 1

    hex_color_dict = {}
    rgb_color_dict = {}
    color_map = plt.get_cmap(color_scale)

    for i in range(num_colors):
        normalized_color = i / (num_colors-1)
        rgb_color = color_map(normalized_color)
        hex_color_dict[i] = matplotlib.colors.rgb2hex(rgb_color[:3])
        rgb_color_dict[i] = rgb_color[:3]

        # print the rgb color
        print(f"Color {i}: {rgb_color}")

    # print the color map
    print(f"Color map for {nx}x{ny}: {color_map}")
    
    print(f"Coloring for {nx}x{ny} (each row with it's color): {ic}")
    print(f"Hex values for {nx}x{ny} (each row with it's color): {[rgb_color_dict[i] for i in ic]}")

    color_ptr = [0 for i in range(num_colors + 1)]
    color_sorted_rows = [0 for i in range(nx*ny)]

    # this sets the color pointer
    for i in range(nx*ny):
        color_ptr[ic[i]+1] += 1
    
    for i in range(num_colors):
        color_ptr[i+1] += color_ptr[i]

    for c in range(num_colors):
        index = color_ptr[c]
        for i in range(nx*ny):
            if ic[i] == c:
                color_sorted_rows[index] = i
                index += 1
    print(f"Coloring for {nx}x{ny} (color pointer): {color_ptr}")
    # print(f"Hex values for {nx}x{ny} (color pointer): {[hex_color_dict[i] for i in color_sorted_rows]}")
    print(f"Coloring for {nx}x{ny} (color sorted rows): {color_sorted_rows}")
    print(f"Hex values for {nx}x{ny} (color sorted rows): {[rgb_color_dict[ic[i]] for i in color_sorted_rows]}")

    # plot an array with the colors

    # Inline 2D plot mimicking an array
    fig, ax = plt.subplots(figsize=(len(ic), 2))

    # Plot the upper row (indices)
    for i, index in enumerate(range(len(ic))):
        ax.text(i, 1, str(index), ha='center', va='center', fontsize=25, color='black')

    # Plot the lower row (colors and ci values)
    for i, color_index in enumerate(ic):
        hex_color = matplotlib.colors.rgb2hex(rgb_color_dict[color_index][:3])  # Get hex color
        # ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1.5, 1, color=hex_color, linewidth=2, edgecolor='black')) 
        # Add colored rectangle
        ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 1, facecolor=hex_color, edgecolor='black', linewidth=3, zorder=0))  # Add colored rectangle with black border
        ax.text(i, 0, str(color_index), ha='center', va='center', fontsize=25, color='white', zorder=10.0)

    # Draw custom borders behind the rectangles
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(ic), 1, edgecolor='black', facecolor='none', linewidth=3, zorder=5))  # Outer border


    # Add labels to the left
    ax.text(-1, 1, "Row", ha='right', va='center', fontsize=25, color='black')
    ax.text(-1, 0, "Color", ha='right', va='center', fontsize=25, color='black')

    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, len(ic) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # save the plot
    plt.tight_layout()
    save_path_figure = save_path + f"2D_propagated_coloringRowColors.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')
    plt.close()

    # now let's do COR format

    # Inline COR format plot
    fig, ax = plt.subplots(figsize=(len(color_sorted_rows), 4))

  # Plot the Color Pointer row
    for i, value in enumerate(color_ptr):
        ax.add_patch(plt.Rectangle((i - 0.5, 2), 1, 1, facecolor='white', edgecolor='black', linewidth=3))  # Black rectangle
        ax.text(i, 3.5, str(i), ha='center', va='center', fontsize=25, color='black')  # Index
        ax.text(i, 2.5, str(value), ha='center', va='center', fontsize=25, color='black')  # Value

    # Plot the Color Sorted Rows
    for i, color_index in enumerate(color_sorted_rows):
        hex_color = matplotlib.colors.rgb2hex(rgb_color_dict[ic[color_index]][:3])  # Get hex color
        ax.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1, facecolor=hex_color, edgecolor='black', linewidth=3))  # Colored rectangle
        ax.text(i, 0.5, str(color_index), ha='center', va='center', fontsize=25, color='white')  # Value

    # Add labels to the left
    ax.text(-1.5, 3.5, "Color", ha='right', va='center', fontsize=25, color='black')
    ax.text(-1.5, 2.5, "Color Pointer", ha='right', va='center', fontsize=25, color='black')
    ax.text(-1.5, 0.5, "Color Sorted Rows", ha='right', va='center', fontsize=25, color='black')

    # Add arrows
    ax.annotate("", xy=(2, 1), xytext=(2, 2), arrowprops=dict(arrowstyle="->", color='black', lw=3))
    ax.annotate("", xy=(4, 1), xytext=(3, 2), arrowprops=dict(arrowstyle="->", color='black', lw=3))
    ax.annotate("", xy=(6, 1), xytext=(4, 2), arrowprops=dict(arrowstyle="->", color='black', lw=3))


    # Set axis limits and remove ticks
    ax.set_xlim(-1.5, len(color_sorted_rows) - 0.5)
    ax.set_ylim(-0.5, 3.5)  # Increased ylim to add whitespace
    ax.set_xticks([])
    ax.set_yticks([])

    # Set aspect ratio to ensure rectangles are squared
    ax.set_aspect('equal', adjustable='box')

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Save the plot
    plt.tight_layout()
    save_path_figure = save_path + f"2D_propagated_coloring_COR_format.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_3D_propagated_coloring(nx, ny, nz):
    # Generate propagated coloring for 3D
    x, y, z, colors = generate_3D_propagated_coloring(nx, ny, nz)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    node_size = 800 if nx < 8 else 400

    # Create the scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap=color_scale, s=node_size, edgecolor='black', alpha=1.0)

    # Add color values as text on the dots
    if(nx < 8):
        for i in range(len(x)):
            ax.text(x[i], y[i], z[i], str(colors[i]), color='white', fontsize=10, ha='center', va='center')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_zlim(-0.5, nz - 0.5)

    # Set ticks
    ax.set_xticks(np.arange(0, nx, 1))
    ax.set_yticks(np.arange(0, ny, 1))
    ax.set_zticks(np.arange(0, nz, 1))

    # Save the plot
    save_path_figure = save_path + f"3D_propagated_coloring_{nx}x{ny}x{nz}.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()

def plot_3D_box_coloring(nx, ny, nz, bx, by, bz):

    # Generate propagated coloring for 3D
    x, y, z, colors = generate_3D_box_coloring(nx, ny, nz, bx, by, bz)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot with opaque nodes
    scatter = ax.scatter(x, y, z, c=colors, cmap=color_scale, s=500, edgecolor='black', alpha=1.0)

    # Add color values as text on the dots
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], str(colors[i]), color='white', fontsize=10, ha='center', va='center')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_zlim(-0.5, nz - 0.5)

    # Set ticks
    ax.set_xticks(np.arange(0, nx, 1))
    ax.set_yticks(np.arange(0, ny, 1))
    ax.set_zticks(np.arange(0, nz, 1))

    # Save the plot
    save_path_figure = save_path + f"3D_box_coloring_{nx}x{ny}x{nz}_{bx}x{by}x{bz}.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()

def plot_min_max_coloring():

    nx = 4
    ny = 4

    x = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    colors = [0, 2, 3, 1, 1, 5, 4, 5, 2, 0, 2, 1, 3, 1, 3, 4]

    # Add pale grey gridlines
    plt.grid(color='lightgrey', linestyle='--', linewidth=1.0, zorder=0.0)

    # Set x and y limits
    plt.xlim(-0.5, nx-0.5)
    plt.ylim(-0.5, ny-0.5)

    # Set ticks
    plt.xticks(np.arange(0, nx, 1))
    plt.yticks(np.arange(0, ny, 1))

    # Remove the enclosing box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    # Create the scatter plot
    plt.scatter(x, y, c=colors, cmap=color_scale, s=800, edgecolor='black', zorder=5.0)

    # Add color values as text on the dots
    for i in range(len(x)):
        plt.text(x[i], y[i], str(colors[i]), color='white', fontsize=15, ha='center', va='center', zorder = 10)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    # plt.title(f'Box Coloring (ColoringBox {bx}x{by})')

    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    save_path_figure = save_path + f"2D_minMax_coloring.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()


# plot_2D_box_coloring(4, 4, 2, 2)
# plot_2D_box_coloring(4, 4, 3, 3)

# plot_2D_propagated_coloring(4, 4)
# plot_2D_propagated_coloring_COR_format(4, 4)
# plot_min_max_coloring()

# plot_3D_propagated_coloring(4, 4, 4)
# plot_3D_propagated_coloring(8, 8, 8)

# these are very "unübersichtlich. so we don't need them in the thesis"
# plot_3D_box_coloring(4, 4, 4, 2, 2, 2)
# plot_3D_box_coloring(4, 4, 4, 3, 3, 3)
# plot_3D_box_coloring(8, 8, 8, 2, 2, 2)
# plot_3D_box_coloring(8, 8, 8, 3, 3, 3)

# plot_3D_propagated_coloring(64, 64, 64)