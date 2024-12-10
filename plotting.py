#################################################################################################################
# Set which data to plot (this is very similar to the settings in get_times.py)
#################################################################################################################
# data_path = "/media/jiela/DK 128GB/Uni/HS24/Masters Thesis/Research - Coding/HighPerformanceHPCG_Thesis/timing_results"
# plot_path = "/media/jiela/DK 128GB/Uni/HS24/Masters Thesis/Research - Coding/HighPerformanceHPCG_Thesis/plots"
data_path = "timing_results/"
plot_path = "plots/"

methods_to_plot = [
    # "CG",
    # "MG",
    "SymGS",
    # "SPMV",
    # "Restriction",
    # "Prolongation",
    # "Dot",
    # "WAXPBY",
]

sizes_to_plot =[
    ("8x8x8"),
    ("16x16x16"),
    # ("24x24x24"),
    ("32x32x32"),
    ("64x64x64"),
    ("128x64x64"),
    ("128x128x64"),
    ("128x128x128"),
    # ("256x128x128"),
    # ("256x256x128"),
]

versions_to_plot = [
    # "BaseTorch",
    # "MatlabReference",
    # "BaseCuPy",
    # "NaiveBanded CuPy",
    # "cuSparse&cuBLAS", #this is a legacy name, it is now called CSR Implementation
    "CSR-Implementation",
    # "Naive Banded",
    # "Naive Banded (1 thread per physical core)",
    # "Naive Banded (4 thread per physical core)",
    # "Banded explicit Shared Memory",
    # "Banded explicit Shared Memory (rows_per_SM pow2)",
    # "Banded explicit Shared Memory (rows_per_SM pow2 1024 threads)",
    # "Banded explicit Shared Memory (rows_per_SM pow2 1024 threads 2x physical cores)",
    "Banded Warp Reduction",
    "Banded Warp Reduction (pre-compute diag_offset)",
    "Banded Warp Reduction (cooperation number = 16)",
    # "Banded Warp Reduction (loop body in method)",
    # "Banded Warp Reduction (8 cooperating threads)",
    # "Banded Warp Reduction - many blocks - 4 threads cooperating",
    # "Banded Warp Reduction - many blocks - 8 threads cooperating",
    "Banded Preprocessed (2 rows while preprocessing)",
    "Banded Preprocessed",
    "Banded Preprocessed (16 rows while preprocessing)",
     
]
plot_percentage_baseline = True

baseline_implementations = [
    "CSR-Implementation",
    ]

y_axis_to_plot = [
    "Time per NNZ (ms)",
    "Time (ms)",
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

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set the display width
pd.set_option('display.max_colwidth', None)  # Set the max column width
#################################################################################################################
# read the data
#################################################################################################################

def read_data():

    # recursively go over all folders and read in all the files
    files = [os.path.join(dp, f) for dp, _, filenames in os.walk(data_path) for f in filenames]

    full_data = pd.DataFrame()

    # print(len(files), flush=True)

    for file in files:
        # the first line contains the metadata, read it in
        with open(file, "r") as f:
            file_content = f.read()

        lines = file_content.splitlines()
        meta_data = lines[0].split(",")

        version_name = str(meta_data[0])
        version_name = "CSR-Implementation" if version_name == "cuSparse&cuBLAS" else version_name # we change the name because I am too lazy to re-run the benchmark
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
    
    return full_data

#################################################################################################################
# preprocess the data
#################################################################################################################
global dense_ops_to_plot
global sparse_ops_to_plot
global cpp_implementation_to_plot
global python_implementation_to_plot

def get_percentage_of_baseline_data(full_data):

    for baseline in baseline_implementations:
        baseline_data = full_data[full_data['Version'] == baseline]
        baseline_medians_ms = baseline_data.groupby(['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'])['Time (ms)'].median().reset_index()
        baseline_medians_nnz = baseline_data.groupby(['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'])['Time per NNZ (ms)'].median().reset_index()

        ms_name = f'Speedup vs {baseline} (Time ms)'
        ms_per_nnz_name = f'Speedup vs {baseline} (Time per NNZ ms)'

        baseline_medians_ms = baseline_medians_ms.rename(columns={'Time (ms)': ms_name})
        baseline_medians_nnz = baseline_medians_nnz.rename(columns={'Time per NNZ (ms)': ms_per_nnz_name})

        y_axis_to_plot.append(ms_name)
        y_axis_to_plot.append(ms_per_nnz_name)

        # Merge the baseline medians with the full data
        full_data = full_data.merge(baseline_medians_ms, on=['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'], how='left')
        full_data = full_data.merge(baseline_medians_nnz, on=['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'], how='left')

        show_data = full_data.drop(columns=['nx', 'ny', 'nz', 'NNZ', 'Density of A', 'Matrix Type', 'Ault Node', 'Matrix Dimensions, Matrix Density'])

        # print(show_data.head())

        # Calculate the speedup compared to the baseline
        full_data[ms_name] = ((full_data['Time (ms)']) / full_data[ms_name])
        full_data[ms_per_nnz_name] = ((full_data['Time per NNZ (ms)']) / full_data[ms_per_nnz_name])


    return full_data

def preprocess_data(full_data):
    global dense_ops_to_plot, sparse_ops_to_plot, cpp_implementation_to_plot, python_implementation_to_plot
    # print(full_data, flush=True)

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

    python_implementations = [
        "BaseTorch",
        "MatlabReference",
        "BaseCuPy",
        "NaiveBanded CuPy"
    ]

    cpp_implementations = [
        "cuSparse&cuBLAS",
        "Naive Banded",
        "Naive Banded (1 thread per physical core)",
        "Naive Banded (4 thread per physical core)",
        "Banded explicit Shared Memory",
        "Banded explicit Shared Memory (rows_per_SM pow2)",
        "Banded explicit Shared Memory (rows_per_SM pow2 1024 threads)",
        "Banded explicit Shared Memory (rows_per_SM pow2 1024 threads 2x physical cores)",
    ]

    all_implementation = python_implementations + cpp_implementations
    for version in all_versions:
        if version not in all_implementation:
            print(f"Version {version} neither in list of cpp implementations nor in list of python implementations", flush=True)

    dense_ops_to_plot = [m for m in methods_to_plot if m in all_dense_ops]
    sparse_ops_to_plot = [m for m in methods_to_plot if m in all_sparse_ops]
    cpp_implementation_to_plot = [v for v in versions_to_plot if v in cpp_implementations]
    python_implementation_to_plot = [v for v in versions_to_plot if v in python_implementations]

    if plot_percentage_baseline:
        full_data = get_percentage_of_baseline_data(full_data)

    # for showing we remove nx,ny,nz, NNZ, Density of A
    show_data = full_data.drop(columns=['nx', 'ny', 'nz', 'NNZ', 'Density of A', 'Matrix Type', 'Ault Node', 'Matrix Dimensions, Matrix Density'])

    # print(show_data.head())
    print(all_matrix_types, flush=True)
    print(all_versions, flush=True)
    print(all_methods, flush=True)
    print(all_ault_nodes, flush=True)
    print(all_matrix_dimensions, flush=True)

    return full_data

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

    # print(len(data), flush=True)

    # if we have no data we do not want to plot anything
    if data.empty:
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(25, 6))

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
    

def plot_x_options(y_axis, y_axis_scale, save_path, full_data):

    # print(len(full_data), flush=True)

    for version in versions_to_plot:

        # filter data
        data = full_data[full_data['Version'] == version]
        # print(len(data), flush=True)
        
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
            # print(dense_data.head())
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

# read in the data
full_data = read_data()
full_data = preprocess_data(full_data)

# print("IN ACTUAL EXECUTION", flush=True)
# print(full_data.head())

print(sparse_ops_to_plot, flush=True)
print(dense_ops_to_plot, flush=True)
print(y_axis_to_plot, flush=True)

for y_ax in y_axis_to_plot:

    # drop all rows with NaN values in the y_ax column
    plottable_data = full_data.dropna(subset=[y_ax])

    if "linear" in y_axis_config_to_plot:
        
        linear_folder = os.path.join(current_plot_path, "y_axis_linear_" + y_ax)
        if not os.path.exists(linear_folder):
            os.makedirs(linear_folder)

        plot_x_options(y_axis = y_ax, y_axis_scale = "linear", save_path = linear_folder, full_data = plottable_data)

    if "log" in y_axis_config_to_plot and "Speedup" not in y_ax:
        # the percentage plots are not done for log scale, because that makes no sense

        log_folder = os.path.join(current_plot_path, "y_axis_log_" + y_ax)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        plot_x_options(y_axis = y_ax, y_axis_scale = "log", save_path = log_folder, full_data = plottable_data)
    


        



