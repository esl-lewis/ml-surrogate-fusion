import numpy as np
import csv
import pandas as pd
import bisect
import sys, os

## Linear interpolation
dir_path = os.path.dirname(os.path.realpath(__file__))

file_list = []
dataframe_list = []

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if "interpolate.py" not in f:
            # ignore self
            full_path = os.path.join(folder, f)
            file_list.append(full_path)

"""
data_series1 = [16, 9, 4, 1, -4, -9, 16]
time_series1 = [10, 20, 30, 40, 50, 60, 70]
time_series2 = [1, 15, 30, 45, 60]
data_series2 = [23, 43, 234, 1, 2]

df1 = pd.DataFrame(list(zip(data_series1, time_series1)), columns=["data", "time"])
df2 = pd.DataFrame(list(zip(data_series2, time_series2)), columns=["data", "time"])


data1 = {'Time': [10, 20, 30, 40, 50, 60, 70], 'data_x': [16, 9, 4, 1, -4, -9, 16],'data_y': [2,44, 2, 1,12,43,1],'data_f':[12,432,54,12,345,5,3]}
data2 = {'Time': [0, 15, 30, 45, 60], 'data_z': [3, 9, 12, 15, 43],'data_i': [2, 44, 2, 1,12]}

df1 = pd.DataFrame.from_dict(data1)
df2 = pd.DataFrame.from_dict(data2)
"""


def interpolate_dataframes(dataframe1, dataframe2):
    """ Where time_series are dataframes, transforming df1 timestep into 2
        returning dataframe2 with interpolated dataframe1 values added"""
    # Figure out if time1 is in range of 2
    time_series1 = dataframe1["Time"]
    time_series2 = dataframe2["Time"]

    max_series2 = time_series2.max()
    min_series2 = time_series2.min()
    # print('df2 max time')
    # print(max_series2)
    # print('df2 min time')
    # print(min_series2)

    def interpolate(timestep1, value1, timestep2, value2, time_to_interp):
        grad = (value1 - value2) / (timestep1 - timestep2)
        interpolated_val = grad * time_to_interp + value1 - timestep1 * grad
        return interpolated_val

    def nearest_values(timestep, timeseries):
        idx = bisect.bisect(timeseries, timestep)
        if 0 < idx < len(timeseries):
            t_below = timeseries[idx - 1]
            t_above = timeseries[idx]
            return (t_below, t_above)
        else:
            raise ValueError("Timestep {} is out of bounds".format(timestep))

    time = []
    data = []
    for times in time_series2:
        print("time value", times)
        # For a given alt timestep, find nearest two timesteps in orig time series
        try:
            time_below, time_above = nearest_values(times, time_series1)
        except ValueError as e:
            print(e, "skipping")
            continue
        time.append(times)
        # print("grabbed nearest times", time_above, time_below)
        above_val = dataframe1.loc[dataframe1["Time"] == time_above]
        below_val = dataframe1.loc[dataframe1["Time"] == time_below]

        above_val = above_val.drop(columns=["Time"])
        below_val = below_val.drop(columns=["Time"])

        # print('values below')
        # print(below_val)

        # print('values above')
        # print(above_val)
        # print(type(above_val))

        interpolated_values = []
        for params in range(0, len(above_val.columns)):
            # print('col index',params)
            # print(below_val.iloc[0,params])
            # print(above_val.iloc[0,params])

            inter_val = interpolate(
                time_above,
                above_val.iloc[0, params],
                time_below,
                below_val.iloc[0, params],
                times,
            )
            # print('calculated val:',inter_val)
            interpolated_values.append(inter_val)
        with_columns = dict(zip(above_val.columns, interpolated_values))
        data.append(with_columns)

    print(time)
    print(data)

    reformat_data = {k: [d.get(k) for d in data] for k in set().union(*data)}

    # Apply to whole dataframe
    time = {"Time": time}
    results = dict(time, **reformat_data)
    print(results)

    # print(results)
    results = pd.DataFrame.from_dict(results)
    results.set_index("Time")
    dataframe2.set_index("Time")
    dataframe2 = dataframe2.astype("float64", errors="ignore")
    results = results.astype("float64", errors="ignore")
    dataframe2 = dataframe2.merge(results, how="outer")
    #dataframe2 = dataframe2.dropna(axis=0)
    print(dataframe2)

    return dataframe2


print(file_list)
for sep_file in file_list:
    # print(sep_file)
    full_filename = os.path.basename(sep_file)
    # filename, file_extension = os.path.splitext(sep_file)
    # print(filename)
    filename, file_extension = os.path.splitext(full_filename)
    # print(filename, file_extension)
    if file_extension == ".csv":
        try:
            pulsenum, dda = filename.split("_")
            print(pulsenum)
            print(dda)
        except:
            print("not a valid file")
            continue
        if dda == "EFIT":
            mag_filename = str(pulsenum) + "_MAGC" + ".csv"
            df_efit = pd.read_csv(full_filename)
            df_magc = pd.read_csv(mag_filename)

            interpolated_df = interpolate_dataframes(df_efit, df_magc)
            interpolated_filename = str(pulsenum) + "_interpolated" + ".csv"
            interpolated_df.to_csv(interpolated_filename)

    # if file_extension == ".csv":
    #    print(filename)


"""
df3 = interpolate_dataframes(df1, df2)


df_mag = pd.read_csv('99070_EFIT.csv')
df_efit = pd.read_csv('99070_MAGC.csv')


# Returns dataframe 1 data in dataframe 2 time series steps
df3 = interpolate_dataframes(df_efit, df_mag)

df3.to_csv('interpolated_99070.csv')
"""
