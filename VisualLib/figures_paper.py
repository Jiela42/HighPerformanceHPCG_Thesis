import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

plot_path = "plots/"

def generate_striped_matrix_visualization():

    n = 8

    matrix = np.zeros((n, n))
    offsets = np.array([-5,-4,-1,0,1,4,5])
    stripe_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'plum', 'lightsalmon', 'lightgray']

    # generate a toy matrix
    for i in range(n):
        for j in offsets:
            if i+j >= 0 and i+j < n:
                matrix[i, (i+j)] = -1
                if i+j == i:
                    matrix[i, (i+j)] = 4
    
    matrix[3, 4] = 0
    matrix[4, 3] = 0
    print(matrix)

    num_stripes = 7
    striped_matrix = np.zeros((n, num_stripes))

    for i in range(n):
        for n_stripe in range(num_stripes):
            j = i + offsets[n_stripe]
            if j >= 0 and j < n:
                striped_matrix[i, n_stripe] = matrix[i, j]

    print(striped_matrix)

    # Plot the standard matrix
    fig, ax = plt.subplots(figsize=(8, 8))
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
    plt.savefig('plots/paper_figures/standard_matrix.png')
    plt.savefig('plots/paper_figures/standard_matrix.eps')
    

    ##########################################

    # Plot the striped matrix
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 11), gridspec_kw={'height_ratios':[1,n]})  # Adjust to stack subfigures vertically
    # fig.suptitle('Striped Representation of a Striped Matrix A', fontsize=25, y=0.95)  # Adjust the vertical position of the title
    
    cell_size = 1 / max(num_stripes, n)

    # Plot the offsets
    ax1.axis('off')
    cell_size = 1 / n  # Ensure cell_size is based on the number of rows in the data table
    table1 = Table(ax1, bbox=[0, 0, 1, cell_size * num_stripes])  # Adjust the height of the offsets to be 1 row

    # Add the offset cells (same size as data cells)
    for i in range(num_stripes):
        cell = table1.add_cell(0, i, cell_size, cell_size * num_stripes, text=str(offsets[i]), loc='center', facecolor=stripe_colors[i])
        cell.get_text().set_fontsize(20)

    # Add borders to the offset table
    for (i, j), cell in table1.get_celld().items():
        cell.set_edgecolor('lightgrey')
    ax1.add_table(table1)

    for cell in table1.get_celld().values():
        cell.get_text().set_fontsize(20)

    ax1.annotate('Offsets', xy=(-0.08, 0.5), xycoords='axes fraction', fontsize=20, ha='center', va='center')

    # Plot the data
    ax2.axis('off')
    table2 = Table(ax2, bbox=[0, 0, n * cell_size, num_stripes * cell_size])  # Adjust the height to match the number of rows
    for (i, j), val in np.ndenumerate(striped_matrix):
        background_color = stripe_colors[j]
        cell = table2.add_cell(i, j, cell_size, cell_size, text=str(int(val)), loc='center', facecolor=background_color)
        cell.get_text().set_fontsize(20)

    # Add borders to the data table
    for (i, j), cell in table2.get_celld().items():
        cell.set_edgecolor('lightgrey')
    ax2.add_table(table2)

    for cell in table2.get_celld().values():
        cell.get_text().set_fontsize(20)

    ax2.annotate('Data', xy=(-0.08, 0.5), xycoords='axes fraction', fontsize=20, ha='center', va='center')
    # Save the figure and display it
    # plt.tight_layout()
    fig.subplots_adjust(hspace=0)  # Adjust the vertical spacing between subfigures
    plt.tight_layout()


    plt.savefig('plots/paper_figures/striped_matrix.png')
    plt.savefig('plots/paper_figures/striped_matrix.eps')
    plt.show()

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
    return  Striped_footprint / CSR_footprint

def memory_footprint(nnz, num_rows, additional_zeros, num_stripes, index_size, value_size):

    CSR_footprint = (nnz * value_size + (nnz + num_rows) * index_size)
    Striped_footprint = (nnz + additional_zeros) * value_size + (num_stripes) * index_size
    striped_based_on_stripes = (num_rows * num_stripes) * value_size + (num_stripes) * index_size
    # print(f"Striped: {Striped_footprint[0]}")
    # print(f"Striped based on stripes: {striped_based_on_stripes[0]}")


    return Striped_footprint / CSR_footprint
    


def generate_memory_footprint_analysis():
    # Generate nx values
    nx = np.linspace(2, 256, 400)  # Generate 400 points between 2 and 1028

    # Calculate y values
    y = memory_footprint_3d27pt(nx, 4, 8)
    y2 = memory_footprint_3d27pt(nx, 4, 4)

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
    plt.figure(figsize=(10, 6))
    plt.plot(nx, y, label=r'float64')
    plt.plot(nx, y2, label=r'float32')
    plt.xlabel('nx')
    plt.ylabel('CSR/Striped memory footprint ratio')
    plt.title('Memory Footprint Ratio')
    plt.legend()
    plt.grid(True)

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


    



if __name__ == "__main__":
    # generate_striped_matrix_visualization()
    generate_memory_footprint_analysis()
