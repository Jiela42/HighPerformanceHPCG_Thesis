import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


def generate_striped_matrix():

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

    # Save the figure and display it
    plt.savefig('plots/paper_figures/standard_matrix.png')
    plt.savefig('plots/paper_figures/standard_matrix.eps')
    

    ##########################################

    # Plot the striped matrix
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios':[1,n]})  # Adjust to stack subfigures vertically
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

    plt.savefig('plots/paper_figures/striped_matrix.png')
    plt.savefig('plots/paper_figures/striped_matrix.eps')
    plt.show()


if __name__ == "__main__":
    generate_striped_matrix()
