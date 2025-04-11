import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.dat', sep=r'\s+')
# Normalize by time/nnz * gpus
data['Runtime'] = (data['Runtime'] / data['NNZ']) * data['GPUs']

# Create the line plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='GPUs', y='Runtime', marker='o', errorbar=('ci', 95)) #make sure we use the same errorbar setting everywhere! 

# Axis labels
plt.xlabel('Number of GPUs')
plt.ylabel('Normalized Runtime [ms]')

# Show the plot
plt.tight_layout()
#plt.show()
plt.savefig("fig10.pdf")
plt.savefig("fig10.png")
