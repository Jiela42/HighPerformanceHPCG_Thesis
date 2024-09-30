import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from io import StringIO
import datetime


#################################################################################################################
# read the data
#################################################################################################################
data_path = "../data/"

# recursively go over all folders and read in all the files
files = [os.path.join(dp, f) for dp, _, filenames in os.walk(data_path) for f in filenames]

full_data = pd.DataFrame()

for file in files:
    # the first line contains the metadata, read it in
    with open(file, "r") as f:
        file_content = f.read()

    lines = file_content.splitlines()
    meta_data = lines[0].split(",")

    version_name = meta_data[0]
    ault_node = meta_data[1]
    matrix_type = meta_data[2]
    nx = int(meta_data[3])
    ny = int(meta_data[4])
    nz = int(meta_data[5])
    nnz = int(meta_data[6])
    method = meta_data[7]

    # read in the rest of the data i.e. the timings

    data = pd.read_csv(StringIO("\n".join(lines[1:])), header=None, names=['Time (ms)'])

    # Add metadata as columns to the data
    data['Version'] = version_name
    data['Ault Node'] = ault_node
    data['Matrix Type'] = matrix_type
    data['nx'] = nx
    data['ny'] = ny
    data['nz'] = nz
    data['NNZ'] = nnz
    data['Method'] = method

    # Append the data to the full_data DataFrame
    full_data = pd.concat([full_data, data], ignore_index=True)

#################################################################################################################
# preprocess the data
#################################################################################################################

# time per nnz
full_data['Time per NNZ (ms)'] = full_data['Time (ms)'] / full_data['NNZ']

# Here we could do preprocessing, such as time per nnz or sorting of the possible values of the columns

# add a column matrix dimensions: nx x ny x nz (a string)
full_data['Matrix Dimensions'] = full_data['nx'].astype(str) + "x" + full_data['ny'].astype(str) + "x" + full_data['nz'].astype(str)


all_matrix_types = full_data['Matrix Type'].unique()
all_versions = full_data['Version'].unique()
all_methods = full_data['Method'].unique()
all_ault_nodes = full_data['Ault Node'].unique()
all_matrix_dimensions = full_data['Matrix Dimensions'].unique()

print(full_data)

#################################################################################################################
# generate the plots
#################################################################################################################

plot_path = "../plots/"

# make new timestamped folder in plots to avoid overwriting old plots
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_plot_path = os.path.join(plot_path, timestamp)


def plot_data(data, x, y, hue, style, title, x_label, y_label, save_path):
    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, hue=hue, style=style, data=data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.show()
