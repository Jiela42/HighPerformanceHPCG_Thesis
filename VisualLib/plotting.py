#################################################################################################################
# Set which data to plot (this is very similar to the settings in get_times.py)
#################################################################################################################
# data_path = "/media/jiela/DK 128GB/Uni/HS24/Masters Thesis/Research - Coding/HighPerformanceHPCG_Thesis/timing_results"
# plot_path = "/media/jiela/DK 128GB/Uni/HS24/Masters Thesis/Research - Coding/HighPerformanceHPCG_Thesis/plots"
data_path = "../timing_results/"
plot_path = "plots/"

methods_to_plot = [
    "CG",
    # "CG_noPreconditioning",
    # "MG",
    # "SymGS",
    # "SPMV",
    # "Restriction",
    # "Prolongation",
    # "Dot",
    # "WAXPBY",
]

sizes_to_plot =[
    # ("8x8x8"),
    # ("16x16x16"),
    # ("24x24x24"),
    ("32x32x32"),
    ("64x64x64"),
    ("128x64x64"),
    ("128x128x64"),
    ("128x128x128"),
    ("256x128x128"),
    # ("256x256x128"),
    ("256x256x256"),
    ("512x512x512"),
    (str(64*8) + "x" + str(64*8) + "x" + str(64*8)),
    (str(128*8) + "x" + str(128*8) + "x" + str(128*8)),
    (str(256*8) + "x" + str(256*8) + "x" + str(256*8)),
    (str(512*8) + "x" + str(512*8) + "x" + str(512*8)),
]

machines_to_plot = [
    "RTX3090",
    "GH200",
    "A100",
    "V100",
]

if len(machines_to_plot) > 1:
    print("Please select only one machine to plot, we plot the first machine in the list.")
machine_to_plot = machines_to_plot[0]


versions_to_plot = [

    ###### Active versions ######
    "AMGX",
    "Striped Multi GPU",
    "Striped coloring (COR Format already stored on the GPU)",
    "Striped coloring (pre-computing COR Format)",
    
    "Striped Box coloring (coloringBox 3x3x3)",
    "Striped Box coloring (coloringBox 2x2x2)",

    "Striped Box coloring (COR stored on GPU) (coloringBox 3x3x3)",

]
plot_percentage_baseline = False
plot_speedup_vs_baseline = False
plot_memory_roofline = False

baseline_implementations = [
    # "AMGX",
    # "Striped coloring (COR Format already stored on the GPU)",
    "Striped Box coloring (coloringBox 3x3x3)",


    # "Striped Box coloring",
    # "Striped Box coloring (COR stored on GPU) (coloringBox 3x3x3)",
    # "Striped Box coloring (coloringBox 3x3x3) (coop_num 4)",
    
    # "Striped Box coloring (changed Norm) (coloringBox 3x3x3) (coop_num 4)",

    # "Striped Box coloring (changed Norm) (coloringBox 3x3x3) (coop_num 4)",
    # "AMGX (converging) (non deterministic)",
    ]

y_axis_to_plot = [
    # "Time per NNZ (ms)",
    "Time (ms)",
]

y_axis_config_to_plot = [
    # "linear",
    "log"
]

make_eps_plots = True
num_measurements_to_skip_for_warm_cache = 2


# Developer options
update_legend_labels = False

#################################################################################################################
# import necessary libraries
#################################################################################################################

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import os
from io import StringIO
from collections import defaultdict
import datetime
import warnings
import re

# Print library versions
print("Library Versions:")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Seaborn: {sns.__version__}")

#################################################################################################################
# Standard L2 Norms for perfectly dependent SymGS
SymGS_L2_Norms = defaultdict(lambda: 0.0, {
    "8x8x8": 30.25627326455036,
    "16x16x16": 59.00057727121956,
    "24x24x24": 100.89687347113858,
    "32x32x32": 155.16087169517397,
    "64x64x64": 461.57263495961604,
    "128x64x64": 663.2354204119574
})

y_L2_Norms = defaultdict(lambda: 1.0, {
    "8x8x8": 175.3396703544295,
    "16x16x16": 336.9391636482764,
    "24x24x24": 500.8392955829245,
    "32x32x32": 667.6466131120565,
    "64x64x64": 1364.858967073155,
    "128x64x64": 1781.8080704722381,
    "128x128x64": 2294.576213595879,
    "128x128x128": 2897.5341240440985,
    "256x128x128": 3828.4398911305893
})

original_CG_num_iterations = defaultdict(lambda: 0, {
    "24x24x24": 23,
    "32x32x32": 29,
    "64x64x64": 55,
    "128x64x64": 71,
    "128x128x64": 85,
    "128x128x128": 104,
    "256x128x128": 137,
    "256x256x128": 165,
    "256x256x256": 189,
    "512x512x512": 367,
})

#################################################################################################################
# Set the Memory Bandwidth for Roofline Plots
memory_bandwidth_GBs = {
    # ault nodes 41-44 have RTX3090s
    "41-44": 936,
    "GH200": 4000,
}
#################################################################################################################

name_map = {
    "Striped coloring (COR Format already stored on the GPU)" : "Striped Propagated Coloring (Precomputed)",
    "Striped coloring (pre-computing COR Format)" : "Striped Propagated Coloring (On-the-fly)",
    "Striped Box coloring (coloringBox 3x3x3)" : "Striped Box Coloring (bx=by=bz=3)",
    "Striped Box coloring (coloringBox 2x2x2)" : "Striped Box Coloring (bx=by=bz=2)",

    "Striped Box coloring (COR stored on GPU) (coloringBox 3x3x3)" : "Striped Box Coloring (bx=by=bz=3) (Precomputed)",
    "Striped Box coloring (COR stored on GPU) (coloringBox 2x2x2)" : "Striped Box Coloring (bx=by=bz=2) (Precomputed)",
}

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

# apply the name map to all possible versions
for version in all_possible_versions:
    if version in name_map:
        new_name = name_map[version]
        all_possible_versions[all_possible_versions.index(version)] = new_name
    else:
        pass

for version in versions_to_plot:
    if version in name_map:
        new_name = name_map[version]
        versions_to_plot[versions_to_plot.index(version)] = new_name
    else:
        pass

for version in baseline_implementations:
    if version in name_map:
        new_name = name_map[version]
        baseline_implementations[baseline_implementations.index(version)] = new_name
    else:
        pass


# Assign unique colors to each implementation
# cmap = plt.cm.get_cmap(sns.color_palette, len(all_possible_versions))

palette = sns.color_palette(sns.color_palette("deep"), len(all_possible_versions))
# Assign unique colors to each implementation
version_colors = {version: palette[i] for i, version in enumerate(all_possible_versions)}

#################################################################################################################
# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set the display width
pd.set_option('display.max_colwidth', None)  # Set the max column width

#################################################################################################################
# Set the theoretical number of bytes to read
#################################################################################################################
# this assumes 3D27pt matrices

def get_theoretical_bytes_read(nx, ny, nz, method):

    # print(f"{nx}x{ny}x{nz}, {method}", flush=True)
    case = method
    if case == "SymGS":
        # First the matrix
        num_doubles = 27 * nx * ny * nz
        num_ints = 27

        # take into account the two vectors
        num_doubles += 2 * nx * ny * nz
        # num_doubles += (27*2 + 1) * nx * ny * nz

        # double both because we have two loops
        num_doubles *= 2
        num_ints *= 2

        return num_doubles * 8 + num_ints * 4
    
    if case == "SPMV":
         # First the matrix
        num_doubles = 27 * nx * ny * nz
        num_ints = 27

        # take into account the two vectors
        num_doubles += 2 * nx * ny * nz

        return num_doubles * 8 + num_ints * 4
    
    if case == "Dot":
        # we have two vectors
        num_doubles = 2*(nx * ny * nz)
        return num_doubles * 8
    
    if case == "WAXPBY":
        # we have three vectors
        num_doubles = 3*(nx * ny * nz)
        return num_doubles * 8

    if case == "MG":

        total_byte = 0

        for depth in range(3):
            total_byte += 2 * get_theoretical_bytes_read(nx, ny, nz, "SymGS")
            total_byte += get_theoretical_bytes_read(nx, ny, nz, "SPMV")

            # adjust nx, ny, nz for the next depth
            nx = nx // 2
            ny = ny // 2
            nz = nz // 2

        return total_byte
    
    if case == "CG":
        total_byte = get_theoretical_bytes_read(nx, ny, nz, "MG")
        total_byte += get_theoretical_bytes_read(nx, ny, nz, "SPMV")
        total_byte += 3 * get_theoretical_bytes_read(nx, ny, nz, "Dot")
        total_byte += 3 * get_theoretical_bytes_read(nx, ny, nz, "WAXPBY")

        num_iterations = original_CG_num_iterations[f"{nx}x{ny}x{nz}"]
        return num_iterations * total_byte
    
    if case == "CG_noPreconditioning":
        return 1
    
    else :
        print(f"Method {method} not found for theoretical bytes", flush=True)
        return 1

#################################################################################################################
# read the data
#################################################################################################################

def read_data():
    # print(f"Reading data from {data_path}", flush=True)

    # recursively go over all folders and read in all the files
    files = [os.path.join(dp, f) for dp, _, filenames in os.walk(data_path) for f in filenames]

    full_data = pd.DataFrame()

    # print(len(files), flush=True)

    version_names = []

    for file in files:
        # print(file, flush=True)
        # the first line contains the metadata, read it in
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            file_content = f.read()

        lines = file_content.splitlines()
        meta_data = lines[0].split(",")

        version_name = str(meta_data[0])
        version_name = name_map[version_name] if version_name in name_map else version_name

        # this portion can help finding the old measurements that we don't need anymore
        # if version_name ==  "Striped coloring (COR Format already stored on the GPU)":
        #     print(f"Found Striped coloring (COR Format already stored on the GPU) in {file}", flush=True)

        # print(f"metadata: {meta_data}", flush=True)
        ault_node = str(meta_data[1])
        matrix_type = str(meta_data[2])
        nx = int(meta_data[3])
        ny = int(meta_data[4])
        nz = int(meta_data[5])
        nnz = int(meta_data[6])
        method = str(meta_data[7])
        sparsity_of_A = nnz / (nx * ny * nz) ** 2 if method not in ["Dot", "WAXPBY"] else 1
        additional_info = str(meta_data[8]) if len(meta_data) > 8 else ""
        # grab the norm


        # Extract NPX, NPY, NPZ from the first header line
        np_match = re.search(r"NPX=(\d+)\s+NPY=(\d+)\s+NPZ=(\d+)", additional_info)
        if np_match:
            file_npx, file_npy, file_npz = map(int, np_match.groups())
            nx = file_npx * nx
            ny = file_npy * ny
            nz = file_npz * nz
        else:
            file_npx = file_npy = file_npz = None

        if(nnz == 0):
            # Replace nnz = int(meta_data[6]) with the following code:
            num_interior_points = (nx - 2) * (ny - 2) * (nz - 2)
            num_face_points = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2))
            num_edge_points = 4 * ((nx - 2) + (ny - 2) + (nz - 2))
            num_corner_points = 8

            nnz_interior = 27 * num_interior_points
            nnz_face = 18 * num_face_points
            nnz_edge = 12 * num_edge_points
            nnz_corner = 8 * num_corner_points

            nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner
        
        # sanity check the numbers
        assert nnz > 0
        assert nx * ny * nz > 0

        # print("Warning for now we ignore any ault_node that is not 41-44", flush=True)
        if ault_node not in machines_to_plot:
            continue

        if version_name not in version_names:
            version_names.append(version_name)
            print(f"Found version {version_name}", flush=True)

        l2_norm_pattern = r"L2 Norm: ([\d\.]+)"
        l2_norm_match = re.search(l2_norm_pattern, additional_info)
        rr_norm_pattern = r"RR Norm: ([\d\.]+)"
        rr_norm_match = re.search(rr_norm_pattern, additional_info)

        if l2_norm_match:
            l2_norm = float(l2_norm_match.group(1))
            # if method == "SymGS" and version_name in versions_to_plot:

            #     print(f"Found L2 Norm: {l2_norm}, {nx}, {version_name}", flush=True)
        else:
            l2_norm = SymGS_L2_Norms[f"{nx}x{ny}x{nz}"]
            # if method == "SymGS" and version_name in versions_to_plot:
            #     print(f"no L2 Norm found, {additional_info}, {version_name}", flush=True)
            #     print(meta_data, flush=True)
        # read in the rest of the data i.e. the timings

        if rr_norm_match:
            rr_norm = float(rr_norm_match.group(1))
            # if rr_norm == 0.2323997:
            #     # print the file name
            #     print("WE FOUND THE RR NORM")
            #     print(file, flush=True)
            #     exit()
        else:

        # assert y_L2_Norms[f"{nx}x{ny}x{nz}"] != 0, f"y_L2_Norms[{nx}x{ny}x{nz}] is 0"


            rr_norm = l2_norm / y_L2_Norms[f"{nx}x{ny}x{nz}"]
            # if(rr_norm == 0.23239969865485122):
                # print("We calculated the rr norm")
                # print(file, flush=True)
                # exit()
        # rr_norm = l2_norm / (nx * ny * nz)

        data = pd.read_csv(StringIO("\n".join(lines[num_measurements_to_skip_for_warm_cache:])), header=None, names=['Time (ms)'])

        # filter out measurements where 'Time (ms)' is 0
        data = data[data['Time (ms)'] != 0]

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

        # If NPX/NPY/NPZ were extracted, add them as well
        if 'file_npx' in locals() and file_npx is not None:
            data['NPX'] = file_npx
            data['NPY'] = file_npy
            data['NPZ'] = file_npz


        # if (version_name in versions_to_plot and method == "SymGS"):
        #     print(f"Read in data for {version_name}, {ault_node}, {matrix_type}, {nx}x{ny}x{nz}, {method}, {additional_info}, {l2_norm}", flush=True)
    

        # only symmetric gauss seidel produces a norm
        if "SymGS" in method:
            data['L2 Norm'] = l2_norm
            data['rr_norm'] = rr_norm
            # print(l2_norm, flush=True)

        # Append the data to the full_data DataFrame
        full_data = pd.concat([full_data, data], ignore_index=True)
    
    # print all versions
    # print(full_data['Version'].unique(), flush=True)
    # print(full_data['Method'].unique(), flush=True)

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

        ms_name = f'Time normalized by {baseline}'
        # ms_per_nnz_name = f'Speedup vs {baseline} (Time per NNZ ms)'

        baseline_medians_ms = baseline_medians_ms.rename(columns={'Time (ms)': ms_name})
        # baseline_medians_nnz = baseline_medians_nnz.rename(columns={'Time per NNZ (ms)': ms_per_nnz_name})

        y_axis_to_plot.append(ms_name)
        # y_axis_to_plot.append(ms_per_nnz_name)

        # Merge the baseline medians with the full data
        full_data = full_data.merge(baseline_medians_ms, on=['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'], how='left')
        # full_data = full_data.merge(baseline_medians_nnz, on=['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'], how='left')

        show_data = full_data.drop(columns=['nx', 'ny', 'nz', 'NNZ', 'Density of A', 'Matrix Type', 'Ault Node', 'NXxNYxNZ = #Rows'])

        # print(show_data.head())

        # Calculate the speedup compared to the baseline
        full_data[ms_name] = ((full_data['Time (ms)']) / full_data[ms_name])
        # full_data[ms_per_nnz_name] = ((full_data['Time per NNZ (ms)']) / full_data[ms_per_nnz_name])


    return full_data

def get_speedup_vs_baseline_data(full_data):
    for baseline in baseline_implementations:
        baseline_data = full_data[full_data['Version'] == baseline]
        baseline_medians_ms = baseline_data.groupby(['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'])['Time (ms)'].median().reset_index()
        # baseline_medians_nnz = baseline_data.groupby(['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'])['Time per NNZ (ms)'].median().reset_index()

        ms_name = f'Speedup vs {baseline}'
        # ms_per_nnz_name = f'Speedup vs {baseline} (Time per NNZ ms)'

        baseline_medians_ms = baseline_medians_ms.rename(columns={'Time (ms)': ms_name})
        # baseline_medians_nnz = baseline_medians_nnz.rename(columns={'Time per NNZ (ms)': ms_per_nnz_name})

        y_axis_to_plot.append(ms_name)
        # y_axis_to_plot.append(ms_per_nnz_name)

        # Merge the baseline medians with the full data
        full_data = full_data.merge(baseline_medians_ms, on=['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'], how='left')
        # full_data = full_data.merge(baseline_medians_nnz, on=['Method', 'Matrix Size', 'Ault Node', 'Matrix Type'], how='left')

        show_data = full_data.drop(columns=['nx', 'ny', 'nz', 'NNZ', 'Density of A', 'Matrix Type', 'Ault Node', 'NXxNYxNZ = #Rows'])

        # print(show_data.head())

        # Calculate the speedup compared to the baseline
        full_data[ms_name] = (full_data[ms_name] / full_data['Time (ms)'])
        # full_data[ms_per_nnz_name] = ((full_data['Time per NNZ (ms)']) / full_data[ms_per_nnz_name])


    return full_data

def get_percentage_of_memBW_data(full_data):

    if(get_theoretical_bytes_read(256, 256, 256, 'CG')) == 0:
       exit()
    col_name = "Percentage of Memory Bandwidth"
    y_axis_to_plot.append(col_name)

    # Calculate theoretical bytes read for each row
    theoretical_bytes_read_B = full_data.apply(lambda row: get_theoretical_bytes_read(row['nx'], row['ny'], row['nz'], row['Method']), axis=1)

    # Get memory bandwidth for each row
    memory_bandwidth = full_data['Ault Node'].map(memory_bandwidth_GBs)

    # Calculate method bandwidth in bytes per millisecond
    method_Bms = theoretical_bytes_read_B / full_data['Time (ms)']



    # print the theoretical bytes read for a specific method and size
    print("Theoretical bytes read for CG 256x256x256: ", flush=True)
    print(theoretical_bytes_read_B[(full_data['Method'] == "CG") & (full_data['Matrix Size'] == "256x256x256")].head(5), flush=True)
    print(f"theoretical bytes read method output: {get_theoretical_bytes_read(256, 256, 256, 'CG')}", flush=True)

    # Convert memory bandwidth to bytes per millisecond
    memory_bandwithd_Bms = memory_bandwidth * 1e6

    # Calculate percentage of memory bandwidth used
    full_data[col_name] = (method_Bms / memory_bandwithd_Bms) * 100

    print("hello there")
    # print all the unique values
    print(f"sizes: {full_data['Matrix Size'].unique()}", flush=True)
    print(f"methods: {full_data['Method'].unique()}", flush=True)
    print(f"versions: {full_data['Version'].unique()}", flush=True)

    # print some values for sizes 256x256x256
    print("256x256x256", flush=True)
    print(full_data[full_data['Matrix Size'] == "256x256x256"][['Method', 'Version', 'Time (ms)', col_name]].sort_values(by=col_name, ascending=False).head(5), flush=True)
    print("512x512x512", flush=True)
    print(full_data[full_data['Matrix Size'] == "512x512x512"][['Method', 'Version', 'Time (ms)', col_name]].sort_values(by=col_name, ascending=False).head(5), flush=True)


    print("256x256x256", flush=True)
    print(full_data[(full_data['Matrix Size'] == "128x128x128") & (full_data['Method'] == "CG")][['Method', 'Version', 'Matrix Size', 'Time (ms)', col_name]].sort_values(by=col_name, ascending=False).head(5), flush=True)

    print(full_data[(full_data['Matrix Size'] == "256x256x256") & (full_data['Method'] == "CG")][['Method', 'Version', 'Matrix Size', 'Time (ms)', col_name]].sort_values(by=col_name, ascending=False).head(5), flush=True)
    return full_data


def preprocess_data(full_data):
    global dense_ops_to_plot, sparse_ops_to_plot, cpp_implementation_to_plot, python_implementation_to_plot
    # print(full_data.head(), flush=True)

    # time per nnz
    full_data['Time per NNZ (ms)'] = full_data['Time (ms)'] / full_data['NNZ']

    # Here we could do preprocessing, such as time per nnz or sorting of the possible values of the columns

    # add a column matrix dimensions: nx x ny x nz (a string)
    full_data['# Rows int'] = full_data['nx'] * full_data['ny'] * full_data['nz']
    full_data['# Rows'] = full_data.apply(
        lambda row: f"{row['nx']}³" if row['nx'] == row['ny'] == row['nz'] else f"{row['nx']}x{row['ny']}x{row['nz']}",
        axis=1
    )
    full_data['NXxNYxNZ = #Rows'] = full_data['nx'].astype(str) + "x" + full_data['ny'].astype(str) + "x" + full_data['nz'].astype(str) + "="  + full_data['# Rows int'].astype(str) # +  ', '+ (full_data['Density of A']* 100).round(2).astype(str) + "%" 
    full_data['Matrix Size'] = full_data['nx'].astype(str) + "x" + full_data['ny'].astype(str) + "x" + full_data['nz'].astype(str)

    # remove any data, that is not to be plotted
    full_data = full_data[full_data['Method'].isin(methods_to_plot)]
    full_data = full_data[full_data['Version'].isin(versions_to_plot)]
    full_data = full_data[full_data['Matrix Size'].isin(sizes_to_plot)]

    all_matrix_types = full_data['Matrix Type'].unique()
    all_versions = full_data['Version'].unique()
    all_methods = full_data['Method'].unique()
    all_ault_nodes = full_data['Ault Node'].unique()
    all_matrix_dimensions = full_data['NXxNYxNZ = #Rows'].unique()

    all_sparse_ops = [
        "SymGS",
        "SPMV",
        "MG",
        "CG",
        "CG_noPreconditioning",
        ]

    all_dense_ops = [
        "Dot",
        "WAXPBY",
        ]

    python_implementations = [
        "BaseTorch",
        "MatlabReference",
        "BaseCuPy",
        "NaiveStriped CuPy"
    ]

    cpp_implementations = [
        "cuSparse&cuBLAS",
        "Naive Striped",
        "Naive Striped (1 thread per physical core)",
        "Naive Striped (4 thread per physical core)",
        "Striped explicit Shared Memory",
        "Striped explicit Shared Memory (rows_per_SM pow2)",
        "Striped explicit Shared Memory (rows_per_SM pow2 1024 threads)",
        "Striped explicit Shared Memory (rows_per_SM pow2 1024 threads 2x physical cores)",
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

    if plot_speedup_vs_baseline:
        full_data = get_speedup_vs_baseline_data(full_data)
    
    if plot_memory_roofline:
        full_data = get_percentage_of_memBW_data(full_data)

    # for showing we remove nx,ny,nz, NNZ, Density of A
    show_data = full_data.drop(columns=['nx', 'ny', 'nz', 'NNZ', 'Density of A', 'Matrix Type', 'Ault Node', 'NXxNYxNZ = #Rows'])

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
    return min(max_cols, num_hues)

def get_legend_horizontal_offset(num_cols, hue_order):
    num_hues = len(hue_order)
    num_rows = num_hues/num_cols

    return -(num_rows + 1) * 0.06 - 0.18

def plot_data(data, x, x_order, y, hue, hue_order, title, save_path, y_ax_scale, color_palette):

    # print(len(data), flush=True)

    # if we have no data we do not want to plot anything
    if data.empty:
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(30, 8))

    # print("hue column values", data[hue].unique(), flush=True)
    # print("hue order", hue_order, flush=True)

    ax = sns.barplot(x=x, order=x_order, y=y, hue=hue, hue_order=hue_order, palette=color_palette, data=data, estimator= np.median, errorbar=("ci", 0.98), err_kws={"alpha": 0.9, "linewidth": 5, "color": "#363535", "linestyle": "-"}, capsize=0.01 )
    fig = ax.get_figure()
    
   # Group the data by the relevant columns
    grouped_data = data.groupby([x, hue])

    # Initialize bar counter
    bar_ctr = 0

    text_size = 30

    # Iterate over x_order and hue_order to annotate bars
    if y != "Percentage of Memory Bandwidth":
        for hue_value in hue_order:
            for x_value in x_order:
                # Get the group corresponding to the current x_value and hue_value
                group_key = (x_value, hue_value)
                if group_key in grouped_data.groups:
                    group_data = grouped_data.get_group(group_key)
                    l2_norms = np.unique(group_data['L2 Norm'].values)
                    rr_norms = np.unique(group_data['rr_norm'].values)
                    v_name = group_data['Version'].values[0]
                    assert len(rr_norms) <= 1, f"More than one rr norm found for a group, this could be because of old data that doesn't contain an rr norm mixed with new data that does, version: {v_name}, group_key: {group_key}, rr_norms: {rr_norms}"
                    assert len(l2_norms) <= 1, f"More than one L2 norm found for a group, this could be because of old data that doesn't contain an L2 norm mixed with new data that does, version: {v_name}, group_key: {group_key},  l2_norms: {l2_norms}"
                    l2_norm = group_data['L2 Norm'].values[0]  # Assuming one L2 norm per group
                    rr_norm = group_data['rr_norm'].values[0]  # Assuming one rr norm per group
                    
                    # Annotate the current bar with the L2 norm
                    bar = ax.patches[bar_ctr]
                    height = bar.get_height()
                    bar_x = bar.get_x()
                    bar_width = bar.get_width()
                    # print(f"Bar {bar_ctr} coordinates: x={bar_x}, width={bar_width}, height={height}", flush=True)
                    # print(f"Annotating Bar {bar_ctr} with L2 Norm: {l2_norm}", flush=True)
                    # print(f"Group Key: {group_key}", flush=True)
                    
                    if not np.isnan(rr_norm):
                        ax.annotate(f'{round(rr_norm, 3)}',
                                    xy=(bar_x + bar_width / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=text_size-4)
                    
                    # Increment the bar counter
                    bar_ctr += 1


    ax.set_title(title, fontsize=text_size+4)
    ax.set_xlabel(x, fontsize=text_size+2)
    ax.set_ylabel(y, fontsize=text_size+2)
    ax.tick_params(axis='both', which='major', labelsize=text_size)

    if y_ax_scale == "log":
        ax.set_yscale('log')
    if y_ax_scale == "linear":
        ax.set_yscale('linear')

    nc = get_num_columns(hue_order=hue_order)
    # print(nc, flush=True)
    nc = 2
    box_offset = get_legend_horizontal_offset(num_cols=nc, hue_order=hue_order)

    if update_legend_labels:
        legend_label_updates = {
            "CuPy (converging) (minres)" : "CuPy MINRES",
            "CuPy (converging) (lsmr)" : "CuPy LSMR",
            "CuPy (converging) (gmres)" : "CuPy GMRES",
            "AMGX (converging) (non deterministic)" : "AMGX",
            "Striped coloring (COR Format already stored on the GPU)": "Striped coloring (propagated dependencies)",
            "Striped Box coloring (coloringBox: 3x3x3) (coop_num: 4)": "Striped Box coloring (direct dependencies)",
            }

        handles, labels = ax.get_legend_handles_labels()
        legend_labels = [legend_label_updates.get(label, label) for label in labels]
        legend = ax.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, box_offset -0.3), ncol=nc, prop={'size': text_size})
    else:
        legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, box_offset - 0.3), ncol=nc, prop={'size': text_size})


    # legend = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, box_offset- 0.05), ncol = nc, prop={'size': text_size}, labels = legend_labels)
    legend.set_title(hue, prop={'size': text_size+2})

    fig.savefig(save_path, bbox_inches='tight')

    if make_eps_plots:
        # save as eps
        eps_path = save_path.replace(".png", ".eps")
        fig.savefig(eps_path, bbox_inches='tight')

    fig.clf()
    plt.close(fig)

def generate_order(current_dim):
    sorted_sizes = {}

    for string in current_dim:

        if "³" in string:
            nx = int(string.split("³")[0])
            ny = nx
            nz = nx

        else:
            nx, ny, nz = [int(x) for x in string.split("x")]

        size_string = f"{nx}x{ny}x{nz}"
        if size_string in sizes_to_plot:
            if size_string not in sorted_sizes:
                sorted_sizes[size_string] = []
            sorted_sizes[size_string].append(string)

    order = []

    for size in sizes_to_plot:
        if size in sorted_sizes:
            sorted_sizes[size] =  sorted(sorted_sizes[size], key=lambda x: x[1])
            order += sorted_sizes[size]
    # print(order, flush=True)
    return order
    

def plot_x_options(y_axis, y_axis_scale, save_path, full_data):

    # print(len(full_data), flush=True)

    axis_info = save_path.split("/")[-1]

    # print(f"Plotting {axis_info}", flush=True)


    # for version in versions_to_plot:

    #     # filter data
    #     data = full_data[full_data['Version'] == version]
    #     # print(len(data), flush=True)
        
    #     # we group by sparse and dense operations 
    #     sparse_data = data[data['Method'].isin(sparse_ops_to_plot)]
    #     dense_data = data[data['Method'].isin(dense_ops_to_plot)]


        
    #     # we might want to sort these in accordance with the sorting instructions!
    #     current_sparse_methods = [m for m in sparse_ops_to_plot if m in data['Method'].unique()]
    #     current_dense_methods = [m for m in dense_ops_to_plot if m in data['Method'].unique()]
    #     current_dense_sizes = [s for s in sizes_to_plot if s in dense_data['Matrix Size'].unique()]
    #     current_sparse_sizes = generate_order(sparse_data['NXxNYxNZ = #Rows'].unique())
    #     current_title = version
    #     current_dense_save_path = os.path.join(save_path, axis_info + "_" + version + "_denseOps_grouped_by_sizes.png")
    #     current_sparse_save_path = os.path.join(save_path, axis_info + "_" + version + "_sparseOps_grouped_by_sizes.png")

    #     color_palette = {version_colors[version]}

    #     if not sparse_data.empty:
    #         plot_data(sparse_data, x = 'NXxNYxNZ = #Rows', x_order = current_sparse_sizes, y = y_axis, hue = 'Method', hue_order = current_sparse_methods, title = current_title, save_path = current_sparse_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)
    #     if not dense_data.empty:
    #         # print(dense_data.head())
    #         plot_data(dense_data, x = 'Matrix Size', x_order = current_dense_sizes, y = y_axis, hue = 'Method', hue_order = current_dense_methods, title = current_title, save_path = current_dense_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)

    #     current_dense_save_path = os.path.join(save_path, axis_info + "_" + version + "_denseOps_grouped_by_methods.png")
    #     current_sparse_save_path = os.path.join(save_path, axis_info + "_" + version + "_sparseOps_grouped_by_methods.png")

    #     if not sparse_data.empty:
    #         plot_data(sparse_data, x = 'Method', x_order = current_sparse_methods, y = y_axis, hue = 'NXxNYxNZ = #Rows', hue_order = current_sparse_sizes, title = current_title, save_path = current_sparse_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)
    #     if not dense_data.empty:
    #         plot_data(dense_data, x = 'Method', x_order = current_dense_methods, y = y_axis, hue = 'Matrix Size', hue_order = current_dense_sizes, title = current_title, save_path = current_dense_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)

    for method in methods_to_plot:

        # filter data
        data = full_data[full_data['Method'] == method]
        
        # we might want to sort these in accordance with the sorting instructions!
        current_versions = [v for v in versions_to_plot if v in data['Version'].unique()]
        current_dense_sizes = [s for s in sizes_to_plot if s in data['Matrix Size'].unique()]
        current_sparse_sizes = generate_order(data['# Rows'].unique())
        current_title = method
        current_save_path = os.path.join(save_path,  axis_info + "_" + method + "_grouped_by_versions.png")

        color_palette = {v: version_colors[v] for v in current_versions}

        if method in sparse_ops_to_plot and not data.empty:
            plot_data(data, x = '# Rows', x_order = current_sparse_sizes, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)
        if method in dense_ops_to_plot and not data.empty:
            plot_data(data, x = 'Matrix Size', x_order = current_dense_sizes, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)

        # careful, the following was not updated to distinguish between sparse and dense operations
        # current_save_path = os.path.join(save_path, method + "_grouped_by_sizes.png")
        # plot_data(data, x = 'Matrix Dimensions', x_order = current_sizes, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_save_path, y_ax_scale = y_axis_scale)
    
    # for size in sizes_to_plot:

    #     # filter data
    #     data = full_data[full_data["Matrix Size"] == size]
        
    #     sparse_data = data[data['Method'].isin(sparse_ops_to_plot)]
    #     dense_data = data[data['Method'].isin(dense_ops_to_plot)]

    #     current_dense_sizes = [s for s in sizes_to_plot if s in dense_data['Matrix Size'].unique()]
    #     current_sparse_sizes = generate_order(sparse_data['NXxNYxNZ = #Rows'].unique())

    #     # we might want to sort these in accordance with the sorting instructions!
    #     current_versions = [v for v in versions_to_plot if v in data['Version'].unique()]
    #     current_dense_methods = [m for m in methods_to_plot if m in dense_data['Method'].unique()]
    #     current_sparse_methods = [m for m in methods_to_plot if m in sparse_data['Method'].unique()]
    #     current_title = "3D Matrix Size, Density: " + size
    #     current_dense_save_path = os.path.join(save_path, axis_info + "_" + size + "_denseOps_grouped_by_versions.png")
    #     current_sparse_save_path = os.path.join(save_path, axis_info + "_" + size + "_sparseOps_grouped_by_versions.png")

    #     color_palette = {version_colors[v] for v in current_versions}

    #     if not sparse_data.empty:
    #         plot_data(dense_data, x = 'Method', x_order = current_dense_methods, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_dense_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)
    #     if not dense_data.empty:
    #         plot_data(sparse_data, x = 'Method', x_order = current_sparse_methods, y = y_axis, hue = 'Version', hue_order = current_versions, title = current_title, save_path = current_sparse_save_path, y_ax_scale = y_axis_scale, color_palette = color_palette)

# read in the data
full_data = read_data()
# print(full_data.head(), flush=True)
full_data = preprocess_data(full_data)

# print("IN ACTUAL EXECUTION", flush=True)
# print(full_data.head())

print(sparse_ops_to_plot, flush=True)
print(dense_ops_to_plot, flush=True)
print(y_axis_to_plot, flush=True)

#######################
# we want to print a specific piece of data
# filter_data = full_data[
#     (full_data['Method'] == "CG") &
#     (full_data['Version'] == "Striped Warp Reduction") &
#     (full_data['Matrix Size'] == "128x128x128")
#     ]
# median_speedup = filter_data['Speedup vs BaseCuPy'].median()
# print(f"Median Speedup vs BaseCuPy for SPMV, Striped Warp Reduction, 128x128x128: {median_speedup}", flush=True)

# filtered_data = full_data[ full_data['Version'] == "AMGX"]
# print(filtered_data, flush=True)

# We for every size we want to print the median of the including convergion of the version without convergion

#######################

for y_ax in y_axis_to_plot:

    # drop all rows with NaN values in the y_ax column
    plottable_data = full_data.dropna(subset=[y_ax])

    if "linear" in y_axis_config_to_plot:
        
        linear_folder = os.path.join(current_plot_path, "y_axis_linear_" + y_ax)
        if not os.path.exists(linear_folder):
            os.makedirs(linear_folder)

        plot_x_options(y_axis = y_ax, y_axis_scale = "linear", save_path = linear_folder, full_data = plottable_data)

    if "log" in y_axis_config_to_plot: # and "Speedup" not in y_ax:
        # the percentage plots are not done for log scale, because that makes no sense

        log_folder = os.path.join(current_plot_path, "y_axis_log_" + y_ax)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        plot_x_options(y_axis = y_ax, y_axis_scale = "log", save_path = log_folder, full_data = plottable_data)
    


        



