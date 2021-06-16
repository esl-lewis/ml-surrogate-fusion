import numpy as np
import csv
import pandas as pd
import os

# select all interpolated csvs

dir_path = os.path.dirname(os.path.realpath(__file__))

file_list = []
dataframe_list = []

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if "interpolate.py" not in f:
            # ignore self
            full_path = os.path.join(folder, f)
            file_list.append(full_path)


# pull out a set of times

for sep_file in file_list:

    full_filename = os.path.basename(sep_file)
    filename, file_extension = os.path.splitext(full_filename)

    if ("interpolated" in filename) and (file_extension == ".csv"):
        print(filename)

        pulse_df = pd.read_csv(sep_file)
        print(pulse_df.head)
        # if pulse_df["Time"]

        time_sampled_df = pulse_df[
            pulse_df["Time"]
            .astype(str)
            .str.contains(r"[0-9]+.25[0-9]+|[0-9]+.75[0-9]+")
        ]
        print(time_sampled_df)

        dataframe_list.append(time_sampled_df)


# write to a final data file
final_df = pd.concat(dataframe_list)


final_df.to_csv("sampled_data.csv", index=False)

