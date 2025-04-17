import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import sys
sys.path.append('/users/dknecht/HighPerformanceHPCG_Thesis')
from Python_HPCGLib.MatrixLib.arbitrary_stencil import generate_stencil_matrix, generate_offsets, Shape
plot_path = "plots/"

def generate_striped_matrix_visualization():

    # we generate a 4x4 2d9pt stencil matrix
    n = 16
    nx = 4
    ny = 4

    matrix = generate_stencil_matrix(2, Shape.SQUARE, 1, [4, 4])

    # from this matrix, generate the striped matrix
    num_stripes = 0
    j_min_i =[]

    # discover the offsets
    for i in range(n):
        for j in range(n):
            if matrix[i,j] != 0:
                curr_offset = j-i
                if curr_offset not in j_min_i:
                    j_min_i.append(curr_offset)
                    num_stripes += 1

    # the offsets are the sorted stripes
    print(f"j_min_i: {j_min_i}")
    offsets = np.sort(j_min_i)

    
    # offsets_2d = generate_offsets(2, 1, Shape.SQUARE)


    print(f"Offsets: {offsets}")
    num_stripes = len(offsets)
    # create the striped matrix
    striped_matrix = np.zeros((n, num_stripes))

    for i in range(n):
        for j in range(n):
            if matrix[i,j] != 0:
                curr_offset = j-i
                stripe_idx = np.where(offsets == curr_offset)[0][0]
                striped_matrix[i, stripe_idx] = matrix[i,j]

    cmap = cm.get_cmap('Pastel1', num_stripes) 
    stripe_colors = [mcolors.rgb2hex(cmap(i)) for i in range(num_stripes)]
    stripe_colors[-1] = '#cccccc'


    # Plot the standard matrix
    fig, ax = plt.subplots(figsize=(n, n))
    ax.axis('off')
    # ax.set_title('Dense Representation of a Striped Matrix A', fontsize=25, pad=20)

    # Create a table with the matrix values
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add cells
    for (i, j), val in np.ndenumerate(matrix):
        background_color_idx = np.where(offsets == j-i)[0]
        background_color = stripe_colors[background_color_idx[0]] if len(background_color_idx) > 0 else 'white'
        if val != 0:
            cell = table.add_cell(i, j, 1/n, 1/n, text=str(int(val)), loc='center', facecolor=background_color)
            cell.get_text().set_fontsize(20)
        else:
            # add dummy text to keep the cell white
            if i == 3 and j == 4:
                # add a zero to the cell
                table.add_cell(i, j, 1/n, 1/n, text='0', loc='center', facecolor=background_color)
            elif i == 4 and j == 3:
                # add a zero to the cell
                table.add_cell(i, j, 1/n, 1/n, text='0', loc='center', facecolor=background_color)
            else:

                table.add_cell(i, j, 1/n, 1/n, text='', loc='center', facecolor='white')
   
   # Set the inner dividers to a faint grey color
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('lightgrey')  # Keep inner borders grey
    
    # Display the table
    ax.add_table(table)

    for cell in table.get_celld().values():
        cell.get_text().set_fontsize(20)

    plt.tight_layout()

    # Save the figure and display it
    plt.savefig('plots/paper_figures/standard_2d9pt_matrix.png')
    plt.savefig('plots/paper_figures/standard_2d9pt_matrix.eps')
    

    ##########################################

  # Plot the striped matrix
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(num_stripes, n + 1), gridspec_kw={'height_ratios': [1, n]}  # Adjust figsize and height ratios
    )

    # Calculate cell size to ensure square cells
    cell_size = 1 / max(n, num_stripes)

    # Plot the offsets (Table 1)
    ax1.axis('off')
    table1 = Table(ax1, bbox=[0, 0, n * cell_size, n*cell_size])  # Adjust bbox for offsets

    # Add the offset cells
    for i in range(num_stripes):
        cell = table1.add_cell(0, i, cell_size, cell_size, text=str(offsets[i]), loc='center', facecolor=stripe_colors[i])
        cell.get_text().set_fontsize(20)

    # Add borders to the offset table
    for (i, j), cell in table1.get_celld().items():
        cell.set_edgecolor('lightgrey')
    ax1.add_table(table1)

    # Annotate the offsets
    ax1.annotate('Offsets', xy=(-0.08, 0.5), xycoords='axes fraction', fontsize=20, ha='center', va='center')

    # Plot the data (Table 2)
    ax2.axis('off')
    table2 = Table(ax2, bbox=[0, 0, n * cell_size, n * cell_size])  # Adjust bbox for the matrix

    # Add the matrix cells
    for (i, j), val in np.ndenumerate(striped_matrix):
        background_color = stripe_colors[j]
        cell = table2.add_cell(i, j, cell_size, cell_size, text=str(int(val)), loc='center', facecolor=background_color)
        cell.get_text().set_fontsize(20)

    # Add borders to the data table
    for (i, j), cell in table2.get_celld().items():
        cell.set_edgecolor('lightgrey')
    ax2.add_table(table2)

    # Annotate the data
    ax2.annotate('Data', xy=(-0.08, 0.5), xycoords='axes fraction', fontsize=20, ha='center', va='center')

    # Adjust layout and save the figure
    fig.subplots_adjust(hspace=0.1)  # Reduce vertical spacing between subplots
    plt.tight_layout()

    plt.savefig('plots/paper_figures/striped_2d9pt_matrix.png')
    plt.savefig('plots/paper_figures/striped_2d9pt_matrix.eps')
    plt.show()

def plot_space2matrix_visualization():

    n = 16
    nx = 4
    ny = 4

    matrix = generate_stencil_matrix(2, Shape.SQUARE, 1, [4, 4])

    # plot the matrix

    fig, ax = plt.subplots(figsize=(n, n))
    ax.axis('off')

    # Create a table with the matrix values
    table = Table(ax, bbox=[0, 0, 1, 1])

    highlight_color = '#A2CFFE'  # Light red color for highlighting

    # Add cells
    for (i, j), val in np.ndenumerate(matrix):

        background_color = highlight_color if (i==5) else 'white'
       
        if val != 0:
            cell = table.add_cell(i, j, 1/n, 1/n, text=str(int(val)), loc='center', facecolor=background_color)
            cell.get_text().set_fontsize(20)
        else:
            # add dummy text to keep the cell white
            if i == 3 and j == 4:
                # add a zero to the cell
                table.add_cell(i, j, 1/n, 1/n, text='0', loc='center', facecolor=background_color)
            elif i == 4 and j == 3:
                # add a zero to the cell
                table.add_cell(i, j, 1/n, 1/n, text='0', loc='center', facecolor=background_color)
            else:
                table.add_cell(i, j, 1/n, 1/n, text='', loc='center', facecolor=background_color)
   
   # Set the inner dividers to a faint grey color
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('lightgrey')  # Keep inner borders grey
    
    # Display the table
    ax.add_table(table)

    for cell in table.get_celld().values():
        cell.get_text().set_fontsize(20)

    plt.tight_layout()

    # Save the figure and display it
    plt.savefig('plots/paper_figures/standard_2d9pt_matrix_withDependencyHighlight.png')
    plt.savefig('plots/paper_figures/standard_2d9pt_matrix_withDependencyHighlight.eps')

    plt.close()
    ##########################################
    ##########################################

    # plot the corresponding 2d space
    # Create a 2D grid
    x = np.arange(nx)
    y = np.arange(ny)

        # Add pale grey gridlines
    plt.grid(color='lightgrey', linestyle='--', linewidth=1.0, zorder=0.0)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # adapt the colors
    colors = np.full((nx, ny), 'white', dtype='<U10')

    for i in range(nx):
        for j in range(ny):
                if i < 3 and j < 3:
                    colors[i,j] = highlight_color

    colors[1,1] = '#1e00ff'

    # print(f"colors: {colors}")
    # Create a scatter plot
    plt.scatter(X, Y, s=800, c=colors.flatten(),edgecolors='black', zorder=5.0)
    
    # Add numbers (x + y * nx) to each point
    for i in range(nx):
        for j in range(ny):
            value = i + j * nx  # Calculate x + y * nx
            plt.text(i, j, str(value), color='black', fontsize=15, ha='center', va='center', zorder=10)


    # add boxes around each bx*by region
    box_buffer = 0.3
    ax = plt.gca()
    rect = patches.FancyBboxPatch(
                 (0 - box_buffer, 0 - box_buffer),
                 3-1 + 2 * box_buffer, 3-1 + 2 * box_buffer,
                 boxstyle="round,pad=0.1, rounding_size=0.3",
                 linewidth=1, edgecolor= highlight_color, facecolor=highlight_color, zorder=-5.0)
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


    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    # plt.title(f'Box Coloring (ColoringBox {bx}x{by})')

    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    save_path_figure = "plots/paper_figures/2D9pt_space_withDependencyHighlight.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()

def plot_SymGS_forward_visualization():

    nx = 4
    ny = 4

    # Create a 2D grid
    x = np.arange(nx)
    y = np.arange(ny)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # adapt the colors
    colors = np.full((nx, ny), 'white', dtype='<U10')


    # colors[1,1] = '#1e00ff'

    # print(f"colors: {colors}")
    # Create a scatter plot
    plt.scatter(X, Y, s=800, c=colors.flatten(),edgecolors='black', zorder=5.0)
    
    # Add numbers (x + y * nx) to each point
    for i in range(nx):
        for j in range(ny):
            value = i + j * nx  # Calculate x + y * nx
            plt.text(i, j, str(value), color='black', fontsize=15, ha='center', va='center', zorder=10)          

    arrow_color = "#b82727"

    # Add red arrows from every node to its smaller neighbors
    arrow_pad = 0.3  # Padding factor to shorten the arrows
    diagonal_arrow_pad = 0.2
    for i in range(nx):
        for j in range(ny):
            # Check neighbors in the grid
            if i > 0:  # Left neighbor
                plt.arrow(i - arrow_pad, j, -1 + 2 * arrow_pad, 0, 
                        head_width=0.1, head_length=0.1, fc=arrow_color, ec=arrow_color, 
                        length_includes_head=True, zorder=5)
            if j > 0:  # Bottom neighbor
                plt.arrow(i, j - arrow_pad, 0, -1 + 2 * arrow_pad, 
                        head_width=0.1, head_length=0.1, fc=arrow_color, ec=arrow_color, 
                        length_includes_head=True, zorder=5)
            if i > 0 and j > 0:  # Bottom-left neighbor
                plt.arrow(i - diagonal_arrow_pad, j - diagonal_arrow_pad, -1 + 2 * diagonal_arrow_pad, -1 + 2 * diagonal_arrow_pad, 
                        head_width=0.1, head_length=0.1, fc=arrow_color, ec=arrow_color, 
                        length_includes_head=True, zorder=5)
            


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


    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y', rotation=0)
    # plt.title(f'Box Coloring (ColoringBox {bx}x{by})')

    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    save_path_figure = "plots/paper_figures/2D9pt_space_withDependencies.png"
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_figure.replace('.png', '.eps'), dpi=300, bbox_inches='tight')

    plt.close()

    

# Define the function
def memory_footprint_3d27pt(nx, index_size, value_size):
    ny = nx
    nz = nx
    num_interior_points = (nx - 2) * (ny - 2) * (nz - 2)
    num_face_points = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2))
    num_edge_points = 4 * ((nx - 2) + (ny - 2) + (nz - 2))
    num_corner_points = 8

    nnz_interior = 27 * num_interior_points
    nnz_face = 18 * num_face_points
    nnz_edge = 12 * num_edge_points
    nnz_corner = 8 * num_corner_points

    nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner

    Striped_footprint = ((27 * nx**3) * value_size)
    CSR_footprint = (nnz * value_size + (nnz + nx**3) * index_size)
    return  CSR_footprint / Striped_footprint

def memory_footprint(nnz, num_rows, additional_zeros, num_stripes, index_size, value_size):

    CSR_footprint = (nnz * value_size + (nnz + num_rows) * index_size)
    Striped_footprint = (nnz + additional_zeros) * value_size + (num_stripes) * index_size
    striped_based_on_stripes = (num_rows * num_stripes) * value_size + (num_stripes) * index_size
    # print(f"Striped: {Striped_footprint[0]}")
    # print(f"Striped based on stripes: {striped_based_on_stripes[0]}")


    return Striped_footprint / CSR_footprint
    
def generate_memory_footprint_analysis():
    # Generate nx values
    nx = np.linspace(2, 128, 400)  # Generate 400 points between 2 and 1028

    # Calculate y values
    y = memory_footprint_3d27pt(nx, 4, 8)
    y2 = memory_footprint_3d27pt(nx, 4, 4)
    y3 = memory_footprint_3d27pt(nx, 8, 8)
    y4 = memory_footprint_3d27pt(nx, 8, 4)

    # Find the value of nx where y equals 1500
    # nx_1500 = nx[np.isclose(y, 1.4, atol=1e-2)]
    # if nx_1500.size > 0:
    #     nx_1500_value = nx_1500[0]
    #     y_1500_value = memory_footprint(nx_1500_value)
    #     print(f'The value of nx where y equals 1500 is approximately: {nx_1500_value}')
    # else:
    #     nx_1500_value = None
    #     print('No value of nx found where y equals 1500.')


    # if nx_1500_value is not None:
    #     plt.plot(nx_1500_value, y_1500_value, 'ro')  # Red circle
    #     plt.annotate(f'nx={nx_1500_value:.2f}', xy=(nx_1500_value, y_1500_value), xytext=(nx_1500_value+50, y_1500_value+500),
    #                  arrowprops=dict(facecolor='black', shrink=0.05))


    # Create the plot
    textsize = 20
    # Set global font size
    plt.rcParams.update({'font.size': textsize})  # Adjust this value as needed

    # # print the last 10 values of y
    # print(f"y: {y[-10:]}")
    # print(f"y2: {y2[-10:]}")
    # print(f"y3: {y3[-10:]}")
    # print(f"y4: {y4[-10:]}")

    # compare y2 and y3


    plt.figure(figsize=(12,8))
    plt.plot(nx, y, label=r'$s_{idx} =$ int, $s_{val} =$ float64', linewidth=3)
    plt.plot(nx, y2, label=r'$s_{idx} =$ int, $s_{val} =$ float32', linewidth=3)
    # plt.plot(nx, y3, label=r'$s_{idx} =$ long, $s_{val} =$ float64')
    plt.plot(nx, y4, label=r'$s_{idx} =$ long, $s_{val} =$ float32', linewidth=3)
    plt.xlabel('nx')
    plt.ylabel('CSR/Striped memory-footprint ratio')
    # plt.title('Memory Footprint Ratio')

    plt.legend(fontsize=textsize, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path + '/paper_figures/memory_footprint_plot.png')
    plt.savefig(plot_path + '/paper_figures/memory_footprint_plot.eps')

    nnz = np.linspace(100, 10000, 100)

    # we make some assumptions:
        # numRows = 1/4 * nnz
        # we do get all num_rows
        # there is an additional 1/4 * nnz zeros
        # we have nnz/50 stripes

    y80 = memory_footprint(nnz, nnz//4, nnz//4, nnz//50, 4, 8)
    y90 = memory_footprint(nnz, nnz//4, nnz//8, nnz//50, 4, 8)
    y95 = memory_footprint(nnz, nnz//4, nnz//16, nnz//50, 4, 8)

    ratio80 =  round((nnz[0]//4)/(nnz[0] + (nnz[0]//4)),4)
    print(f"Ratio 80: {ratio80}")
    ratio90 = round((nnz[0]//8)/(nnz[0] + (nnz[0]//8)),4)
    print(f"Ratio 90: {ratio90}")
    ratio95 = round((nnz[0]//16)/(nnz[0] + (nnz[0]//16)),4)
    print(f"Ratio 95: {ratio95}")


    

    y80_100 = memory_footprint(nnz, nnz//4, nnz//4, nnz//100, 4, 8)
    y90_100 = memory_footprint(nnz, nnz//4, nnz//8, nnz//100, 4, 8)
    y95_100 = memory_footprint(nnz, nnz//4, nnz//16, nnz//100, 4, 8)


    plt.figure(figsize=(10, 6))
    plt.plot(nnz, y80, label=f'{ratio80 *100}% zeros')
    # plt.plot(nnz, y80_100, label=f'{ratio80 *100}% zeros, (nnz/100 stripes)')
    plt.plot(nnz, y90, label=f'{ratio90 *100}% zeros')
    # plt.plot(nnz, y90_100, label=f'{ratio90 *100}% zeros, (nnz/100 stripes)')
    plt.plot(nnz, y95, label=f'{ratio95 *100}% zeros')
    # plt.plot(nnz, y95_100, label=f'{ratio95 *100}% zeros, (nnz/100 stripes)')

    plt.xlabel('nnz')
    plt.ylabel('CSR/Striped memory footprint ratio')
    plt.title('Memory Footprint Ratio')
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='lower center', ncol=3)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path + '/paper_figures/memory_footprint_general.png')
    plt.savefig(plot_path + '/paper_figures/memory_footprint_general.eps')

def nnz(nx):
    # Calculate nnz for a 3D 27-point stencil
    num_interior_points = (nx - 2) * (nx - 2) * (nx - 2)
    num_face_points = 2 * ((nx - 2) * (nx - 2) + (nx - 2) * (nx - 2) + (nx - 2) * (nx - 2))
    num_edge_points = 4 * ((nx - 2) + (nx - 2) + (nx - 2))
    num_corner_points = 8

    nnz_interior = 27 * num_interior_points
    nnz_face = 18 * num_face_points
    nnz_edge = 12 * num_edge_points
    nnz_corner = 8 * num_corner_points

    nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner

    return nnz

def plot_sparsity_curve():
    
    # Generate nx values
    nx = np.linspace(2, 1024, 1024)  # Generate 400 points between 2 and 1028
    matrix_size = (nx * nx * nx)**2
    nnzs = nnz(nx)

    density = nnzs / matrix_size * 100

    # print the last 10 values of density
    print(f"density: {density[-10:]}")

    # Create the plot
    textsize = 20
    # Set global font size
    plt.rcParams.update({'font.size': textsize})  # Adjust this value as needed

    plt.figure(figsize=(12,8))
    plt.plot(nx, density, linewidth=3)

    plt.xlabel('nx')
    plt.ylabel('Matrix Density in %')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0, 600)
    # Set x-axis ticks to powers of two
    powers_of_two = 2**np.arange(1, 10)
    print(f"powers_of_two: {powers_of_two}")
    plt.xticks(powers_of_two, labels=[f"{x}" for x in powers_of_two])
    # plt.title('Memory Footprint Ratio')

    plt.legend(fontsize=textsize)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path + '/paper_figures/sparsity_curve.png')
    plt.savefig(plot_path + '/paper_figures/sparsity_curve.eps')

if __name__ == "__main__":
    # generate_striped_matrix_visualization()
    # plot_space2matrix_visualization()
    plot_SymGS_forward_visualization()
    # generate_memory_footprint_analysis()
    # plot_sparsity_curve()
