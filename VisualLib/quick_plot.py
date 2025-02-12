import matplotlib.pyplot as plt
import numpy as np

plot_path = "plots/"

# Define the function
def memory_footprint(nx):
    return (41 * nx**3 - 9 * nx**2) / (27 * nx**3)


# Generate nx values
nx = np.linspace(2, 256, 400)  # Generate 400 points between 2 and 1028

# Calculate y values
y = memory_footprint(nx)

# Find the value of nx where y equals 1500
nx_1500 = nx[np.isclose(y, 1.5, atol=1e-2)]
if nx_1500.size > 0:
    nx_1500_value = nx_1500[0]
    y_1500_value = memory_footprint(nx_1500_value)
    print(f'The value of nx where y equals 1500 is approximately: {nx_1500_value}')
else:
    nx_1500_value = None
    print('No value of nx found where y equals 1500.')


if nx_1500_value is not None:
    plt.plot(nx_1500_value, y_1500_value, 'ro')  # Red circle
    plt.annotate(f'nx={nx_1500_value:.2f}', xy=(nx_1500_value, y_1500_value), xytext=(nx_1500_value+50, y_1500_value+500),
                 arrowprops=dict(facecolor='black', shrink=0.05))


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(nx, y, label=r'$\frac{41nx^3 - 9nx^2}{27nx^3}$')
plt.xlabel('nx')
plt.ylabel('CSR/Striped memory footprint ratio')
plt.title('Memory Footprint Ratio')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(plot_path + 'memory_footprint_plot.png')

# Show the plot
# plt.show()