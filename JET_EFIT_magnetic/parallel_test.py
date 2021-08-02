# EFIT + magnetics

try:
    import multiprocessing as mp

    from numpy.core.arrayprint import DatetimeFormat

    import sys, traceback
    import numpy as np
    import csv
    import time
    import pandas as pd

    # import py_flush as Flush
    from scipy import optimize

    # from math import pi, sin, cos, sqrt, exp, atan2, tanh, cosh
    # import gtk
    import matplotlib
    import matplotlib.pyplot as plt

    # from matplotlib.widgets import Slider
    # from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
    # from matplotlib.figure import Figure
    # from matplotlib.figure import Axes
    # from matplotlib.backends.backend_gtkagg import (
    #    NavigationToolbar2GTK as NavigationToolbar,
    # )
    import os
except:
    print("-------failed to import module!---------")
    traceback.print_exc(file=sys.stdout)
    sys.exit(127)


# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *


# --- data functions
class DATA:
    # Initialise global variables
    def __init__(
        self, EFIT_parameters, MAGC_parameters, flux_probes, magnetic_probes
    ):  # MACHINE DEPENDENT
        # DATA.pulse = 82631
        # DATA.pulse = pulse_number
        DATA.pulse = 0
        DATA.EFIT_t = np.array([])
        DATA.MAGC_t = np.array([])
        DATA.EFIT_x = np.array([])
        DATA.MAGC_x = np.array([])
        DATA.MAGC_mag = magnetic_probes
        DATA.MAGC_flux = flux_probes
        DATA.EFIT_params = EFIT_parameters
        DATA.MAGC_params = MAGC_parameters
        for param in EFIT_parameters:
            setattr(DATA, param, np.array([]))
        for param in MAGC_parameters:
            setattr(DATA, param, np.array([]))
        for probe in magnetic_probes:
            probe_name = "BPME_" + str(probe)
            setattr(DATA, probe_name, np.array([]))
        for probe in flux_probes:
            probe_name = "FLME_" + str(probe)
            setattr(DATA, probe_name, np.array([]))

    # --- Load the pulse basic data # MACHINE DEPENDENT
    def set_pulse(self, pulse_number):
        DATA.pulse = pulse_number

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")  # for disruption database, change to chain1
        ier = ppfgo(pulse=DATA.pulse, seq=0)

        # --- Use MSTA to find which diagnostics are usable
        # check magnetic probe status
        dda = "MSTA"
        dtyp = "STBP"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise IOError(
                "Failed to load {} data. May not exist for pulse.".format(dtyp)
            )
        for probe_number in self.MAGC_mag:
            if data[probe_number] == 1:
                continue
            elif data[probe_number] == 0:
                raise IOError(
                    "Data not found for magnetic probe {}, probably broken. Aborting.".format(
                        probe_number
                    )
                )

        # check flux probe status
        dtyp = "STFL"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise IOError(
                "Failed to load {} data. May not exist for pulse.".format(dtyp)
            )
        for probe_number in self.MAGC_flux:
            if data[probe_number] == 1:
                continue
            elif data[probe_number] == 0:
                raise IOError(
                    "Data not found for flux probe {}, probably broken. Aborting.".format(
                        probe_number
                    )
                )

        # --- Load MAGC data
        dda = "MAGC"
        for param in self.MAGC_params:
            dtyp = param
            ihdat, iwdat, data, x, t, ier = ppfget(
                DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
            )
            if ier != 0:
                raise IOError(
                    "Failed to load {} data. May not exist for pulse.".format(dtyp)
                )

            # filter magnetic probes
            if param == "BPME":
                probe_indices = self.MAGC_mag
                for probe in probe_indices:
                    print("BPME PROBE NUM", probe)
                    probe_name = "BPME_" + str(probe)
                    # probe = probe - 1  # accounting for indexing from zero
                    # TODO replace this hard coding 1061 with len(t)
                    this_probe_indices = list(
                        range(1061 * probe, (1061 * probe) + 1061)
                    )
                    this_probe_values = np.take(data, this_probe_indices)
                    setattr(DATA, probe_name, this_probe_values)

            # filter flux probes
            elif param == "FLME":
                probe_indices = self.MAGC_flux
                for probe in probe_indices:
                    print("FLME PROBE NUM", probe)
                    probe_name = "FLME_" + str(probe)
                    # probe = probe - 1  # accounting for indexing from zero
                    # TODO replace this hard coding 1061 with len(t)
                    this_probe_indices = list(
                        range(1061 * probe, (1061 * probe) + 1061)
                    )
                    this_probe_values = np.take(data, this_probe_indices)
                    setattr(DATA, probe_name, this_probe_values)

            else:
                setattr(DATA, param, data)
        DATA.MAGC_t = t
        DATA.MAGC_x = x

        # Switching over to chain1 EFIT so comparison to MAGC is valid
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("chain1", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        # --- Load EFIT data # MACHINE DEPENDENT
        dda = "EFIT"
        for param in self.EFIT_params:
            dtyp = param
            ihdat, iwdat, data, x, t, ier = ppfget(
                DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
            )
            if ier != 0:
                raise IOError(
                    "Failed to load {} data. May not exist for pulse.".format(dtyp)
                )
            # DATA.EFIT_xip = data
            if (param == "P") | (param == "DFDP"):
                print("P/DFDP check", data.shape)
                data = np.reshape(data, (33, -1))
                print("P/DFDP check", data.shape)
                # data = data[:, 0]  # check this, could be data[0,:]
                data = data[0, :]
                print("P/DFDP check", data.shape)
            setattr(DATA, param, data)
        DATA.EFIT_t = t
        DATA.EFIT_x = x

        print("timing:")
        print(len(self.MAGC_t))
        print(len(self.EFIT_t))

        return self

    def write_pulse(self, pulse_number):
        EFIT_data = {}
        MAGC_data = {}
        # params_to_retrieve = EFIT_params + MAGC_params
        for parameter in self.EFIT_params:
            EFIT_data[parameter] = getattr(self, parameter)
        EFIT_data["Time"] = DATA.EFIT_t

        all_params = dir(self)

        print("ALL PARAMS(?)", all_params)

        only_probes = filter(
            lambda param: (param.startswith("BPME_")) | (param.startswith("FLME_")),
            all_params,
        )
        filtered_params = list(only_probes)

        for parameter in self.MAGC_params:
            if parameter.startswith("BPME") | parameter.startswith("FLME"):
                continue
            filtered_params.append(parameter)
        print("filtered params:", filtered_params)

        for parameter in filtered_params:
            MAGC_data[parameter] = getattr(self, parameter)
        MAGC_data["Time"] = DATA.MAGC_t

        # for key, value in all_data.items():
        #    print(key, len([item for item in value if item]))

        df_EFIT = pd.DataFrame(EFIT_data)
        # df_EFIT = df_EFIT.set_index("Time")
        df_MAGC = pd.DataFrame(MAGC_data)

        merged_df = df_MAGC.merge(df_EFIT, how="outer", on="Time")
        merged_df = merged_df.dropna(axis=0)

        merged_name = str(pulse_number) + ".csv"
        merged_df["FBND-FAXS"] = merged_df["FBND"] - merged_df["FAXS"]
        merged_df.to_csv(merged_name, index=False)
        print(merged_df.head(2))

    def set_and_write(self, pulse_number):
        try:
            self = self.set_pulse(pulse_number)
            self.write_pulse(pulse_number)
        except Exception as e:
            print("Data for", pulse_number, "not found. Possibly dry run, skipping.")
            print(e)


# Main function to run whole thing
class Main:
    def __init__(self):
        EFIT_params = ["FAXS", "FBND", "P", "DFDP"]  # BOTH
        MAGC_params = ["BPME", "FLME", "BVAC", "FLX", "IPLA"]

        magnetic_probes = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
        ]
        flux_loops = [
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            37,
            38,
        ]

        # params_to_retrieve = input("EFIT params requested:")
        # pulse_num = input("Pulse number:")
        data_thread = DATA(EFIT_params, MAGC_params, flux_loops, magnetic_probes)

        num_processors = mp.cpu_count()

        pool = mp.Pool(num_processors)

        pulse_numbers = list(range(99070, 99072))
        # divide pulses between cores
        # divided_pulses = []

        # with mp.Pool(processes=num_processors) as pool:

        # divide pulses up

        pool.map(data_thread.set_and_write, [pulse_num for pulse_num in pulse_numbers])
        pool.close()

        """for pulse_num in range(99070, 99072):
            try:
                data_thread.set_and_write(pulse_num)
            except Exception as e:
                print("Data for", pulse_num, "not found. Possibly dry run, skipping.")
                print(e)
                continue"""


Main()
