import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_possible_versions = [
    "AMGX",
    "Striped Multi GPU",
    "Striped coloring (COR Format already stored on the GPU)",
    "Striped coloring (pre-computing COR Format)",
    "Striped Box coloring (coloringBox 3x3x3)",
    "Striped Box coloring (coloringBox 2x2x2)",
    "Striped Box coloring (COR stored on GPU) (coloringBox 3x3x3)",
    "Striped Box coloring (COR stored on GPU) (coloringBox 2x2x2)",
]

# we give each implementation its own color
palette = sns.color_palette(sns.color_palette("deep"), len(all_possible_versions))
# Assign unique colors to each implementation
version_colors = {version: palette[i] for i, version in enumerate(all_possible_versions)}


# Load the data
data = pd.read_csv('data.dat', sep=r'\s+')
# Normalize by time/nnz * gpus
data['Runtime'] = ((data['Runtime'] / data['NNZ']) * data['GPUs']) * 1e6

# Create the line plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='GPUs', color=version_colors["Striped Multi GPU"], y='Runtime', marker='o', errorbar=('ci', 98)) #make sure we use the same errorbar setting everywhere! 

# Axis labels
plt.xlabel('Number of GPUs')
plt.ylabel('Normalized Runtime [ns]')

# Show the plot
plt.tight_layout()
#plt.show()
plt.savefig("fig10.pdf")
plt.savefig("fig10.png")
