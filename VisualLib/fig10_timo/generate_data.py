import pandas as pd
import numpy as np
import os
import re
from io import StringIO

measurement_path = "../Figure_weak_scaling/measurements_per_CG_call"
num_iterations_path = "../Figure_weak_scaling/number_iterations_per_NPX_NPY_NPZ.csv"

num_measurements_to_skip_for_warm_cache = 2

files = [os.path.join(dp, f) for dp, _, filenames in os.walk(measurement_path) for f in filenames]

full_data = pd.DataFrame()

version_names = []

for file in files:

    # the first line contains the metadata
    with open(file, "r", encoding="utf-8", errors="replace") as f:
        file_content = f.read()

    lines = file_content.splitlines()
    meta_data = lines[0].split(",")

    version_name = str(meta_data[0])
    version_name = "CSR-Implementation" if version_name == "cuSparse&cuBLAS" or version_name == "cuSparse-Implementation" else version_name # we change the name because I am too lazy to re-run the benchmark
    version_name = version_name.replace("Banded", "Striped")

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
    np_match = re.search(r"NPX=(\d+)\s+NPY=(\d+)\s+NPZ=(\d+)", str(meta_data[8]))
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
    if ault_node not in ["GH200"]:
        continue

    if version_name not in version_names:
        version_names.append(version_name)
        print(f"Found version {version_name}", flush=True)

    data = pd.read_csv(StringIO("\n".join(lines[num_measurements_to_skip_for_warm_cache:])), header=None, names=['Time (ms)'])

    # filter out measurements where 'Time (ms)' is 0
    data = data[data['Time (ms)'] != 0]

    # Add metadata as columns to the data
    data['Version'] = version_name
    data['Ault Node'] = ault_node
    data['Matrix Type'] = matrix_type
    data['NNZ'] = nnz
    data['Method'] = method
    data['Density of A'] = sparsity_of_A

    # If NPX/NPY/NPZ were extracted, add them as well
    if 'file_npx' in locals() and file_npx is not None:
        data['NPX'] = file_npx
        data['NPY'] = file_npy
        data['NPZ'] = file_npz

    nx = int(nx) * file_npx if file_npx is not None else nx
    ny = int(ny) * file_npy if file_npy is not None else ny
    nz = int(nz) * file_npz if file_npz is not None else nz

    # Append the data to the full_data DataFrame
    full_data = pd.concat([full_data, data], ignore_index=True)

# readin the number iterations for each size
num_iterations = pd.read_csv(
    num_iterations_path,
    header=0,  # Use the first row as the header
    skipinitialspace=True,  # Skip spaces after commas
)
num_iterations = num_iterations.loc[:, ~num_iterations.columns.str.contains('^Unnamed')]
print(num_iterations, flush=True)

num_iterations = num_iterations.rename(columns={'#iterations': 'num_iterations'})

full_data = full_data.merge(num_iterations, on=['NPX', 'NPY', 'NPZ'], how='left')
full_data['Runtime per iteration (ms)'] = full_data['Time (ms)'] / full_data['num_iterations']
full_data['GPUs'] = full_data['NPX'] * full_data['NPY'] * full_data['NPZ']

# write runtime & nnz & gpus to file
# make a new pd with only the relevant columns NNZ    GPUs   Runtime
data_to_write = full_data[['NNZ', 'GPUs', 'Runtime per iteration (ms)']]
data_to_write = data_to_write.rename(columns={'Runtime per iteration (ms)': 'Runtime'})
data_to_write = data_to_write.rename(columns={'nnz': 'NNZ'})

print(data_to_write, flush=True)

# Write the data to a .dat file with space-separated values
data_to_write.to_csv('data.dat', sep=' ', index=False, header=True)





