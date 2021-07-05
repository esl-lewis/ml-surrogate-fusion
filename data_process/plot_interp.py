import pandas as pd

import csv
import matplotlib.pyplot as plt
from pandas.core.reshape.merge import merge


# plotting the interp MAGC against normal MAGC and EFIT to get an idea of problem

MAGC_file = "../JET_EFIT_magnetic/99071_MAGC.csv"
EFIT_file = "../JET_EFIT_magnetic/99071_EFIT.csv"
# INTERP_file = "./99071_interpolated.csv"

"""
interp_df = pd.read_csv(INTERP_file)
"""
# interp_plot = interp_df[["Time", "BPME_19"]]
# interp_plot = interp_plot.set_index(["Time"])

mag_df = pd.read_csv(MAGC_file)
# mag_plot = mag_df[["Time", "BPME_19"]]
# mag_plot = mag_plot.set_index(["Time"])

# merge_df = interp_plot.merge(mag_plot, how="outer")

# print(interp_plot.head(4))
# print(mag_plot.head(4))
# print(merge_df.head(4))
# plt.plot(merge_df["Time"], merge_df["BPME_19"])
"""
plt.scatter(
    interp_df["Time"],
    interp_df["BPME_19"],
    color="green",
    marker=".",
    label="Probe 19 interpolated",
)
"""
for column_val in mag_df.columns:
    if column_val.startswith("FLME"):
        plt.plot(
            mag_df["Time"], mag_df[column_val], marker="x", label=str(column_val),
        )
    else:
        continue
"""
plt.plot(
    mag_df["Time"], mag_df["BPME_19"], color="red", marker="x", label="Probe 19 raw",
)
plt.plot(
    mag_df["Time"], mag_df["BPME_11"], color="blue", marker="x", label="Probe 11 raw",
)
plt.plot(
    mag_df["Time"], mag_df["BPME_14"], color="black", marker="x", label="Probe 14 raw",
)
"""
"""
plt.scatter(
    interp_df["Time"],
    interp_df["BPME_27"],
    color="black",
    marker=".",
    label="Probe 27 interpolated",
)
plt.scatter(
    mag_df["Time"], mag_df["BPME_27"], color="blue", marker="x", label="Probe 27 raw"
)
"""
# plt.legend(loc="center right")
plt.xlabel("Time /s")
plt.ylabel("Flux probe output (Wb?)")
plt.show()
