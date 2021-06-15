from numpy.core.arrayprint import DatetimeFormat

# EFIT + magnetics

try:
    import sys, traceback
    import numpy as np
    import csv
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

# --- Global variables
mu_0 = 4 * 3.1415926535 * 1e-7
eV2Joules = 1.602176487 * 1e-19
scale_height = 50

# pulse_number = input("Pulse number:")
# pulse_number = 82630

# TODO add way to check for diagnostic failure, eg EFIT has failed time slices value EFL
# TODO add flag to check for dry run?

# TODO grabs MSTA correct probes
# TODO interpolates times

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
            setattr(DATA, param, data)
        DATA.EFIT_t = t
        DATA.EFIT_x = x

        print("timing:")
        print(len(self.MAGC_t))
        print(len(self.EFIT_t))

        return self


# Main function to run whole thing
class Main:
    def __init__(self):
        # pulse_num = 86320
        EFIT_params = ["FAXS", "FBND"]  # BOTH
        MAGC_params = ["BPME", "FLME", "BVAC", "FLX", "IPLA"]

        magnetic_probes = [
            0,
            3,
            4,
            5,
            6,
            8,
            9,
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
            21,
            22,
            23,
            27,
            30,
            32,
            34,
        ]
        flux_loops = [
            4,
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
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            37,
        ]

        # params_to_retrieve = input("EFIT params requested:")
        # pulse_num = input("Pulse number:")
        data_thread = DATA(EFIT_params, MAGC_params, flux_loops, magnetic_probes)

        # Extract multiple pulses
        for pulse_num in range(99070, 99072):
            try:
                data_thread = data_thread.set_pulse(pulse_num)
            except Exception as e:
                print("Data for", pulse_num, "not found. Possibly dry run, skipping.")
                print(e)
                continue
            EFIT_data = {}
            MAGC_data = {}
            # params_to_retrieve = EFIT_params + MAGC_params
            for parameter in EFIT_params:
                EFIT_data[parameter] = getattr(data_thread, parameter)
            EFIT_data["Time"] = DATA.EFIT_t

            all_params = dir(data_thread)
            only_probes = filter(
                lambda param: (param.startswith("BPME_")) | (param.startswith("FLME_")),
                all_params,
            )
            filtered_params = list(only_probes)

            for parameter in filtered_params:
                MAGC_data[parameter] = getattr(data_thread, parameter)
            MAGC_data["Time"] = DATA.MAGC_t

            # for key, value in all_data.items():
            #    print(key, len([item for item in value if item]))

            df_EFIT = pd.DataFrame(EFIT_data)
            # df_EFIT = df_EFIT.set_index("Time")
            filename = str(pulse_num) + "_EFIT.csv"
            with open(filename, mode="w") as f:
                df_EFIT.to_csv(f)

            df_MAGC = pd.DataFrame(MAGC_data)
            # df_MAGC = df_MAGC.set_index("Time")
            filename = str(pulse_num) + "_MAGC.csv"
            with open(filename, mode="w") as f:
                df_MAGC.to_csv(f)


#    gtk_thread = gtk_class(data_thread)
#    gtk.main()

Main()
