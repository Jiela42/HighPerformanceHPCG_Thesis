#################################################################################################################
# Set which data to plot (this is very similar to the settings in get_times.py)
#################################################################################################################
# data_path = "/media/jiela/DK 128GB/Uni/HS24/Masters Thesis/Research - Coding/HighPerformanceHPCG_Thesis/timing_results"
# plot_path = "/media/jiela/DK 128GB/Uni/HS24/Masters Thesis/Research - Coding/HighPerformanceHPCG_Thesis/plots"
data_path = "../timing_results/"
plot_path = "../plots/"

methods_to_plot = [
    # "CG",
    "MG",
    "SymGS",
    "SPMV",
    # "Restriction",
    # "Prolongation",
    "Dot",
    "WAXPBY",
]

sizes_to_plot =[
    ("8x8x8"),
    ("16x16x16"),
    ("32x32x32"),
    ("64x64x64"),
    ("128x128x128"),
]

versions_to_plot = [
    "cuSparse&cuBLAS",
    "naiveBanded"
]

y_axis_to_plot = [
    "Time per NNZ (ms)",
    "Time (ms)"
]

y_axis_config_to_plot = [
    "linear",
    "log"
]


#################################################################################################################
# import necessary libraries
#################################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from io import StringIO
import datetime
import warnings

#################################################################################################################
# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#################################################################################################################
# read the data
#################################################################################################################

# recursively go over all folders and read in all the files
files = [os.path.join(dp, f) for dp, _, filenames in os.walk(data_path) for f in filenames]

full_data = pd.DataFrame()

for file in files:
    # the first line contains the metadata, read it in
    with open(file, "r") as f:
        file_content = f.read()

    lines = file_content.splitlines()
    meta_data = lines[0].split(",")

    version_name = str(meta_data[0])
    ault_node = str(meta_data[1])
    matrix_type = str(meta_data[2])
    nx = int(meta_data[3])
    ny = int(meta_data[4])
    nz = int(meta_data[5])
    nnz = int(meta_data[6])
    method = str(meta_data[7])
    sparsity_of_A = nnz / (nx * ny * nz) ** 2 if method not in ["Dot", "WAXPBY"] else 1

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
    data['Density of A'] = sparsity_of_A

    # Append the data to the full_data DataFrame
    full_data = pd.concat([full_data, data], ignore_index=True)

#################################################################################################################
# preprocess the data
#################################################################################################################

print(full_data)

# time per nnz
full_data['Time per NNZ (ms)'] = full_data['Time (ms)'] / full_data['NNZ']

# Here we could do preprocessing, such as time per nnz or sorting of the possible values of the columns

# add a column matrix dimensions: nx x ny x nz (a string)
full_data['Matrix Dimensions, Matrix Density'] = full_data['nx'].astype(str) + "x" + full_data['ny'].astype(str) + "x" + full_data['nz'].astype(str) + ", " + (full_data['Density of A']* 100).round(2).astype(str) + "%" 
full_data['Matrix Size'] = full_data['nx'].astype(str) + "x" + full_data['ny'].astype(str) + "x" + full_data['nz'].astype(str)

# remove any data, that is not to be plotted
full_data = full_data[full_data['Method'].isin(methods_to_plot)]
full_data = full_data[full_data['Version'].isin(versions_to_plot)]
full_data = full_data[full_data['Matrix Size'].isin(sizes_to_plot)]

all_matrix_types = full_data['Matrix Type'].unique()
all_versions = full_data['Version'].unique()
all_methods = full_data['Method'].unique()
all_ault_nodes = full_data['Ault Node'].unique()
all_matrix_dimensions = full_data['Matrix Dimensions, Matrix Density'].unique()

all_sparse_ops = [
    "SymGS",
    "SPMV",
    "MG",
    "CG",
    ]

all_dense_ops = [
    "Dot",
    "WAXPBY",
    ]

dense_ops_to_plot = [m for m in methods_to_plot if m in all_dense_ops]
sparse_ops_to_plot = [m for m in methods_to_plot if m in all_sparse_ops]


# print(full_data)
print(all_matrix_types)
print(all_versions)
print(all_methods)
print(all_ault_nodes)
print(all_matrix_dimensions)

#################################################################################################################
# generate the plots
#################################################################################################################

# make new timestamped folder in plots to avoid overwriting old plots
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_plot_path = os.path.join(plot_path, timestamp)

if not os.path.exists(current_plot_path):
        os.makedirs(current_plot_path)

def get_num_columns(hue_order):
    
    longest_name = max([len(h) for h in hue_order])
    num_hues = len(hue_order)

    max_cols = 4 if longest_name <= 10 else 3
    return max(max_cols, num_hues)

def get_legend_horizontal_offset(num_cols, hue_order):
    num_hues = len(hue_order)
    num_rows = num_hues/num_cols

    return -(num_rows + 1) * 0.06 - 0.18

def plot_data(data, x, x_order, y, hue, hue_order, title, save_path, y_ax_scale):

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(x=x, order=x_order, y=y, hue=hue, hue_order=hue_order, data=data,  estimator= np.median, ci=98)
    fig = ax.get_figure()

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if y_ax_scale == "log":
        ax.set_yscale('log')
    if y_ax_scale == "linear":
        ax.set_yscale('linear')

    nc = get_num_columns(hue_order=hue_order)
    box_offset = get_legend_horizontal_offset(num_cols=nc, hue_order=hue_order)

    legend = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, box_offset), ncol = nc)
    legend.set_title(hue)

    fig.savefig(save_path, bbox_inches='tight')

    fig.clf()
    plt.close(fig)

def generate_order(current_dim_perc):
    sorted_sizes = {}

    for string in current_dim_perc:
        size = string.split(",")[0]
        if size in sizes_to_plot:
            if size not in sorted_sizes:
                sorted_sizes[size] = []
            sorted_sizes[size].append(string)
    
    # sort the order from smallest size to largest and sort the percentages within the same size
    
    order = []

    for size in sizes_to_plot:
        if size in sorted_sizes:
            sorted_sizes[size] = sorted(sorted_sizes[size], key=lambda x: float(x.split(",")[1].strip("%")))
            order += sorted_sizes[size]
    return order
    

def plot_x_options(y_axis, y_axis_scale, save_path):

    for version in versions_to_plot:

        # filter data
        data = full_data[full_data['Version'] == version]
        
        # we group by sparse and dense operations 
        sparse_data = data[data['Method'].isin(sparse_ops_to_plot)]
        dense_data = data[data['Method'].isin(dense_ops_to_plot)]
        
        # we might want to sort these in accordance with the sorting instructions!
        current_sparse_methods = [m for m in sparse_ops_to_plot if m in data['Method'].unique()]
        current_dense_methods = [m for m in dense_ops_to_plot if m in data['Method'].unique()]
        current_dense_sizes = [s for s in sizes_to_plot if s in dense_data['Matrix Size'].unique()]
        current_sparse_sizes = generate_order(sparse_data['Matrix Dimensions, Matrix Density'].unique())
        current_title = version
        current_dense_save_path = os.path.join(save_path, version + "_denseOps_grouped_by_sizes.png")
        current_sparse_save_path = os.path.join(save_path, version + "_sparseOps_grouped_by_sizes.png")

        if not sparse_data.empty:
            plot_data(sparse_data, x = 'Matrix Dimensions, Matrix Density', x_order = current_sparse_sizes, y = y_axis, hue = 'Method', hue_order = current_sparse_methods, title = current_title, save_path = current_sparse_save_path, y_ax_scale = y_axis_scale)
        if not dense_data.empty:
            plot_data(dense_data, x = 'Matrix Size', x_order = current_dense_sizes, y = y_axis, hue = 'Method', hue_order = current_dense_methods, title = current_title, save_path = current_dense_save_path, y_ax_scale = y_axis_scale)

        current_dense_save_path = os.path.join(save_path, version + "_denseOps_grouped_by_methods.png")
        current_sparse_save_path = os.path.join(save_path, version + "_sparseOps_grouped_by_methods.png")

        if not sparse_data.empty:
            plot_data(sparse_data, x = 'Method', x_order = current_sparse_methods, y = y_axis, hue = 'Matrix Dimensions, Matrix Density', hue_order = current_sparse_sizes, title = current_title, save_path = current_sparse_save_path, y_ax_scale = y_axis_scale)
        if not dense_data.empty:
            plot_data(dense_data, x = 'Method', x_order = current_dense_methods, y = y_axis, hue = 'Matrix Size', hue_order = current_dense_sizes, title = current_title, save_path = current_dense_save_path, y_ax_scale = y_axis_scale)


    for method in methods_to_plot:

        # filter data
        data = full_data[full_data['Method'] == method]
        
        # we might want to sort these in accordance with the sorting instructions!
        current_versions = [v for v in versions_to_plot if v in data['Version'].unique()]
        current_dense_sizes = [s for s in sizes_to_plot if s in data['Matrix Size'].unique()]
        current_sparse_sizes = generate_order(data['Matrix Dimensions, Matrix Density'].unique())
        current_title = method
        current_save_path = os.path.join(save_path, method + "_grouped_by_versions.png")

        if method in sparse_ops_to_plot and not data.empty:
            plot_data(data, x = 'Matrix Dimensions, Matrix Density', x_order = current_sparse_sizes, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_save_path, y_ax_scale = y_axis_scale)
        if method in dense_ops_to_plot and not data.empty:
            plot_data(data, x = 'Matrix Size', x_order = current_dense_sizes, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_save_path, y_ax_scale = y_axis_scale)

        # careful, the following was not updated to distinguish between sparse and dense operations
        # current_save_path = os.path.join(save_path, method + "_grouped_by_sizes.png")
        # plot_data(data, x = 'Matrix Dimensions', x_order = current_sizes, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_save_path, y_ax_scale = y_axis_scale)
    
    for size in sizes_to_plot:

        # filter data
        data = full_data[full_data["Matrix Size"] == size]
        
        sparse_data = data[data['Method'].isin(sparse_ops_to_plot)]
        dense_data = data[data['Method'].isin(dense_ops_to_plot)]

        current_dense_sizes = [s for s in sizes_to_plot if s in dense_data['Matrix Size'].unique()]
        current_sparse_sizes = generate_order(sparse_data['Matrix Dimensions, Matrix Density'].unique())

        # we might want to sort these in accordance with the sorting instructions!
        current_versions = [v for v in versions_to_plot if v in data['Version'].unique()]
        current_dense_methods = [m for m in methods_to_plot if m in dense_data['Method'].unique()]
        current_sparse_methods = [m for m in methods_to_plot if m in sparse_data['Method'].unique()]
        current_title = "3D Matrix Size, Density: " + size
        current_dense_save_path = os.path.join(save_path, size + "_denseOps_grouped_by_versions.png")
        current_sparse_save_path = os.path.join(save_path, size + "_sparseOps_grouped_by_versions.png")

        if not sparse_data.empty:
            plot_data(dense_data, x = 'Method', x_order = current_dense_methods, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_dense_save_path, y_ax_scale = y_axis_scale)
        if not dense_data.empty:
            plot_data(sparse_data, x = 'Method', x_order = current_sparse_methods, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_sparse_save_path, y_ax_scale = y_axis_scale)

for y_ax in y_axis_to_plot:
    if "linear" in y_axis_config_to_plot:
        
        linear_folder = os.path.join(current_plot_path, "y_axis_linear_" + y_ax)
        if not os.path.exists(linear_folder):
            os.makedirs(linear_folder)

        plot_x_options(y_axis = y_ax, y_axis_scale = "linear", save_path = linear_folder)

    if "log" in y_axis_config_to_plot:

        log_folder = os.path.join(current_plot_path, "y_axis_log_" + y_ax)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        plot_x_options(y_axis = y_ax, y_axis_scale = "log", save_path = log_folder)


        



