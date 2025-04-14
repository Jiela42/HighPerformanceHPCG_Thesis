
data_path = "../profiling_results"


make_eps_plots = False
sizes_to_plot =[
    ("32x32x32"),
    ("64x64x64"),
    ("128x128x128"),
    ("256x256x256"),
    ("512x512x512"),
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
    "AMGX",
    "Striped Box coloring (coloringBox 2x2x2)",
    "Striped Box coloring (coloringBox 3x3x3)",
]

relevant_ncu_metrics = [
    "Duration",
    "Memory Throughput",
    "DRAM Throughput"]

memory_bandwidth_GBs = {
    # ault nodes 41-44 have RTX3090s
    "RTX3090": 936,
    "GH200": 4000,
    "A100": 1555,
    "V100": 900,
}

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os


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

# Assign unique colors to each implementation
# cmap = plt.cm.get_cmap(sns.color_palette, len(all_possible_versions))

palette = sns.color_palette(sns.color_palette("deep"), len(all_possible_versions))
# Assign unique colors to each implementation
version_colors = {version: palette[i] for i, version in enumerate(all_possible_versions)}


def adjust_throughput_metric(df):
    """
    Adjusts the throughput metrics in the DataFrame to ensure they are in Gbyte/second.
    Scales the values and updates the units if necessary.
    Raises an error if any metric is still not in Gbyte/second after adjustment.
    """
    # Adjust "Memory Throughput" to Gbyte/second if the unit is "Mbyte/second"
    if "Memory Throughput_unit" in df.columns:
        df.loc[df["Memory Throughput_unit"] == "Mbyte/second", "Memory Throughput"] /= 1000
        df.loc[df["Memory Throughput_unit"] == "Mbyte/second", "Memory Throughput_unit"] = "Gbyte/second"

    # Check if any metric still does not have the correct unit
    incorrect_units = df[
        (df["Memory Throughput_unit"] != "Gbyte/second")        
    ]


    if not incorrect_units.empty:
        print("Incorrect units found in the following rows:")
        print(incorrect_units)
        raise ValueError("Some metrics still do not have the correct unit (should be Gbyte/second) after adjustment.")

    return df
    
def adjust_duration_metric(df):

    if "Duration_unit" in df.columns:
        # Convert "Duration" to seconds if the unit is "s"
        df.loc[df["Duration_unit"] == "msecond", "Duration"] /= 1000
        df.loc[df["Duration_unit"] == "msecond", "Duration_unit"] = "seconds"

        # Convert "Duration" to seconds if the unit is "nseconds"
        df.loc[df["Duration_unit"] == "nsecond", "Duration"] /= 1e9
        df.loc[df["Duration_unit"] == "nsecond", "Duration_unit"] = "seconds"

        # Convert "Duration" to seconds if the unit is "useconds"
        df.loc[df["Duration_unit"] == "usecond", "Duration"] /= 1e6
        df.loc[df["Duration_unit"] == "usecond", "Duration_unit"] = "seconds"

    # Check if any metric still does not have the correct unit
    incorrect_units = df[
        (df["Duration_unit"] != "seconds")]
    
    if not incorrect_units.empty:
        print("Incorrect units found in the following rows:")
        print(incorrect_units)
        raise ValueError("Some metrics still do not have the correct unit (should be seconds) after adjustment.")
    return df
    


def read_data():

    files = [os.path.join(dp, f) for dp, _, filenames in os.walk(data_path) for f in filenames if f.endswith('.csv')]

    full_data = []

    for file in files:
        
        with open (file, 'r') as f:
            file_content = f.read()
        
        lines = file_content.splitlines()

        # this is an example metadata:
        # AMGX,RTX3090,32x32x32,SymGS

        metadata = lines[0].split(",")

        version_name = metadata[0]
        machine = metadata[1]
        size = metadata[2]
        method = metadata[3]

        if machine != machine_to_plot:
            continue

        size_split = size.split("x")
        nx = int(size_split[0])
        ny = int(size_split[1])
        nz = int(size_split[2])

        df = pd.read_csv(file, skiprows=1)

        # Filter the DataFrame to include only relevant metrics
        df = df[df["Metric Name"].isin(relevant_ncu_metrics)]

        # Further filter for "Memory Throughput" where "Section Name" is "Memory Workload Analysis"
        df = df[
            ~((df["Metric Name"] == "Memory Throughput") & (df["Section Name"] != "Memory Workload Analysis"))
        ]

        # make sure the Metric Value is numeric
        df["Metric Value"] = pd.to_numeric(df["Metric Value"], errors="coerce")

        # make sure there are no duplicates otherwise we need to manually select the right one
        duplicates = df[df.duplicated(subset=["ID", "Metric Name"], keep=False)]
        if not duplicates.empty:
            print("Duplicates found in the following rows:")
            print(duplicates)
            raise ValueError("Duplicates found in the DataFrame. Please check the data.")

        # Pivot the DataFrame to make "Metric Name" values into columns
        df = df.pivot_table(
            index=["ID"],  # Use "ID" (Kernel ID) as the index
            columns="Metric Name",  # Use "Metric Name" values as columns
            values=["Metric Value", "Metric Unit"],  # Use "Metric Value" as the data
            aggfunc="first"  # Use the first value if duplicates exist
        ).reset_index()

        # Flatten the multi-level columns
        df.columns = [
            f"{col[1]}_unit" if col[0] == "Metric Unit" else col[1]
            for col in df.columns
        ]

        # rename the id column to ID
        df.rename(columns={"": "Kernel ID"}, inplace=True)

        # Show all columns and rows when printing the DataFrame
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_rows', None)     # Show all rows (optional)
        pd.set_option('display.width', 1000)       # Adjust the width to avoid line wrapping
        
        # adjust the metrics
        df = adjust_throughput_metric(df)
        df = adjust_duration_metric(df)

        # add bytes transferred
        df["Bytes Transferred in Kernel"] = df["Memory Throughput"] * df["Duration"]

        # sum over all the bytes transferred
        df["Total Bytes Transferred"] = df["Bytes Transferred in Kernel"].sum()
        df["Total Duration"] = df["Duration"].sum()
        # calculate the memory bandwidth
        df["Memory Bandwidth GB/s (kernel)"] = df["Bytes Transferred in Kernel"] / df["Duration"]
        df["Total Memory Bandwidth GB/s"] = df["Total Bytes Transferred"] / df["Total Duration"]

        # now the memory bandwidth utilization
        df["Memory Bandwidth Utilization (kernel)"] = df["Memory Bandwidth GB/s (kernel)"] / memory_bandwidth_GBs[machine]
        df["Total Memory Bandwidth Utilization"] = df["Total Memory Bandwidth GB/s"] / memory_bandwidth_GBs[machine]

        # add metadata to the dataframe
        df["Version"] = version_name
        df["Machine"] = machine
        df["Size"] = size
        df["nx"] = nx
        df["ny"] = ny
        df["nz"] = nz
        df["Method"] = method
        df["Memory Bandwidth GB/s"] = memory_bandwidth_GBs[machine]

        full_data.append(df)


    full_data = pd.concat(full_data, ignore_index=True)
    # preprocess the data
    full_data['NXxNYxNZ = #Rows'] = (
        full_data['nx'].astype(str) + "x" +
        full_data['ny'].astype(str) + "x" +
        full_data['nz'].astype(str) + "="  +
        (full_data["nx"]*full_data["ny"]*full_data["nz"]).astype(str))
    
    full_data['# Rows'] = full_data.apply(
        lambda row: f"{row['nx']}³" if row['nx'] == row['ny'] == row['nz'] else f"{row['nx']}x{row['ny']}x{row['nz']}",
        axis=1
    )

    return full_data



def get_legend_horizontal_offset(num_cols, hue_order):
    num_hues = len(hue_order)
    num_rows = num_hues/num_cols

    return -(num_rows + 1) * 0.06 - 0.18


def plot_data(data, x, x_order, y, hue, hue_order, title, save_path, color_palette):

    # if we have no data we do not want to plot anything
    if data.empty:
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(30, 8))

    ax = sns.barplot(x=x, order=x_order, y=y, hue=hue, hue_order=hue_order, data=data, color_palette=color_palette, estimator= np.median,  errorbar=("ci", 0.98), err_kws={"alpha": 0.9, "linewidth": 5, "color": "#363535", "linestyle": "-"}, capsize=0.01)
    fig = ax.get_figure()

   # Group the data by the relevant columns
    grouped_data = data.groupby([x, hue])

 
    text_size = 30


    ax.set_title(title, fontsize=text_size+4)
    ax.set_xlabel(x, fontsize=text_size+2)
    ax.set_ylabel(y, fontsize=text_size+2)
    ax.tick_params(axis='both', which='major', labelsize=text_size-5)

    nc = 2
    box_offset = get_legend_horizontal_offset(num_cols=nc, hue_order=hue_order)

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, box_offset - 0.2), ncol=nc, prop={'size': text_size})


    # legend = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, box_offset- 0.05), ncol = nc, prop={'size': text_size}, labels = legend_labels)
    legend.set_title(hue, prop={'size': text_size+2})

    fig.savefig(save_path, bbox_inches='tight')

    if make_eps_plots:
        # save as eps
        eps_path = save_path.replace(".png", ".eps")
        fig.savefig(eps_path, bbox_inches='tight')

    fig.clf()
    plt.close(fig)
      

full_data = read_data()
# print the unique values in the dataframe
print("Methods: ", full_data["Method"].unique())
print("Machines: ", full_data["Machine"].unique())
print("Sizes: ", full_data["Size"].unique())
print("Versions: ", full_data["Version"].unique())


# make new timestamped directory
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join("plots", timestamp), exist_ok=True)
save_path = os.path.join("plots", timestamp)

# now let's plot the data

for m in full_data["Method"].unique():
    data = full_data[full_data["Method"] == m]
    data = data[data["Size"].isin(sizes_to_plot)]
    data = data[data["Version"].isin(versions_to_plot)]

    # sort the data by size
    data.sort_values(by=["nx", "ny", "nz"], inplace=True)

    # get the x order
    x_order = data["# Rows"].unique()

    # get the hue order
    hue_order = data["Version"].unique()

    # get the color palette
    color_palette = [version_colors[version] for version in hue_order]

    # now we can plot the data
    for metric in ["Memory Bandwidth Utilization (kernel)", "Total Memory Bandwidth Utilization"]:
        y = metric

        # for each size print the total memory bandwidth utilization
        for size in sizes_to_plot:
            size_data = data[data["Size"] == size]
            for version in versions_to_plot:
                version_data = size_data[size_data["Version"] == version]
                if not version_data.empty:
                    # print(version)
                    print(f"{m} {metric} {size} {version}: {version_data[y].values[0]}")
        # if we have no data we do not want to plot anything
        if data.empty:
            continue

        image_safepath = os.path.join(save_path, f"{m}_{metric}.png")
        plot_data(data, x="# Rows", x_order=x_order, y=y, hue="Version", hue_order=hue_order, title=f"{m} {metric}", save_path=image_safepath, color_palette=color_palette)
