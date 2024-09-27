import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


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
        meta_data = f.readline().split(",")

        version_name = meta_data[0]
        ault_node = meta_data[1]
        matrix_type = meta_data[2]
        nx = meta_data[3]
        ny = meta_data[4]
        nz = meta_data[5]
        method = meta_data[6]

        # read in the rest of the data i.e. the timings

        data = pd.read_csv(file, skiprows=1, header=None)

        # Add metadata as columns to the data
        data['Version'] = version_name
        data['Ault Node'] = ault_node
        data['Matrix Type'] = matrix_type
        data['nx'] = nx
        data['ny'] = ny
        data['nz'] = nz
        data['Method'] = method

        # Append the data to the full_data DataFrame
        full_data = full_data.append(data, ignore_index=True)


    full_data = full_data.append(data)
print(full_data)