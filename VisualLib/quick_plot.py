import matplotlib.pyplot as plt
import numpy as np

plot_path = "plots/"

# Define the function
def memory_footprint(nx, index_size, value_size):
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
    return  (nnz * value_size + (nnz + nx**3) * index_size)/ ((27 * nx**3) * value_size)




# Generate nx values
nx = np.linspace(2, 256, 400)  # Generate 400 points between 2 and 1028

# Calculate y values
y = memory_footprint(nx, 4, 8)
y2 = memory_footprint(nx, 4, 4)

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
plt.savefig(plot_path + 'memory_footprint_plot.png')

# Show the plot
# plt.show()