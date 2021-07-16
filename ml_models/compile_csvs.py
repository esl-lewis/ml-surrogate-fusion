# Merge all the csvs into one giant csv?

import numpy as np
import csv
import pandas as pd
import sys, os

dir_path = os.path.dirname(os.path.realpath(__file__))

file_list = []
dataframe_list = []

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if "interpolate.py" not in f:
            # ignore self
            full_path = os.path.join(folder, f)
            file_list.append(full_path)

for sep_file in file_list:
    # print(sep_file)
    full_filename = os.path.basename(sep_file)
    # filename, file_extension = os.path.splitext(sep_file)
    # print(filename)
    filename, file_extension = os.path.splitext(full_filename)
    # print(filename, file_extension)
    if ("EFIT" in filename) & (file_extension == ".csv"):
        try:
            pulsenum, dda = filename.split("_")
            print("PULSE NUMBER", pulsenum)
            print("DDA", dda)
        except:
            print("not a valid file")
            continue
        mag_filename = str(pulsenum) + "_MAGC" + ".csv"
        df_efit = pd.read_csv(full_filename, index_col=0)
        df_magc = pd.read_csv(mag_filename, index_col=0)

        merged_df = df_magc.merge(df_efit, how="outer", on="Time")
        merged_df = merged_df.dropna(axis=0)
        print(merged_df.head(2))
        print(merged_df.shape)
        dataframe_list.append(merged_df)

final_df = pd.concat(dataframe_list)


final_filename = "all_data.csv"
final_df.to_csv(final_filename, index=False)

