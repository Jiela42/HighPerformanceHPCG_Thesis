import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

save_path = "plots/paper_figures/"

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
    plt.scatter(x, y, c=colors, cmap='tab10', s=800, edgecolor='black', zorder=5.0)

    # Add color values as text on the dots
    for i in range(len(x)):
        plt.text(x[i], y[i], str(colors[i]), color='white', fontsize=15, ha='center', va='center', zorder = 10)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    # plt.title(f'Box Coloring (ColoringBox {bx}x{by})')

    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    save_path_figure = save_path + f"box_coloring_{bx}x{by}.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()


    
# Example usage
plot_2D_box_coloring(4, 4, 2, 2)
plot_2D_box_coloring(4, 4, 3, 3)