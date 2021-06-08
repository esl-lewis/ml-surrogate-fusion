from numpy.core.arrayprint import DatetimeFormat


try:
    import sys, traceback
    import numpy as np
    import csv
    import pandas as pd

    import py_flush as Flush
    from scipy import optimize

    # from math import pi, sin, cos, sqrt, exp, atan2, tanh, cosh
    import gtk
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.figure import Axes
    from matplotlib.backends.backend_gtkagg import (
        NavigationToolbar2GTK as NavigationToolbar,
    )
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
# TODO grabs MAGC times
# TODO grabs MSTA correct probes
# TODO interpolates times

# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *

# --- data functions
class DATA:
    # Initialise global variables
    def __init__(self, EFIT_parameters, MAGC_parameters):  # MACHINE DEPENDENT
        # DATA.pulse = 82631
        # DATA.pulse = pulse_number
        DATA.pulse = 0
        DATA.EFIT_t = np.array([])
        DATA.MAGC_t = np.array([])
        DATA.EFIT_x = np.array([])
        DATA.MAGC_x = np.array([])
        DATA.psi_shift = 0.0
        DATA.EFIT_params = EFIT_parameters
        DATA.MAGC_params = MAGC_parameters
        for param in EFIT_parameters:
            setattr(DATA, param, np.array([]))
        for param in MAGC_parameters:
            setattr(DATA, param, np.array([]))

    # --- Load the pulse basic data # MACHINE DEPENDENT
    def set_pulse(self, pulse_number):
        DATA.pulse = pulse_number

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)

        # --- Load EFIT data # MACHINE DEPENDENT
        dda = "EFIT"
        for param in self.EFIT_params:
            dtyp = str(param)
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
        """
        # --- Use MSTA to find which diagnostics are usable
        # Select a set of working probes
        
        dda = "MSTA"
        dtyp = "EFUS"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise IOError(
                "Failed to load {} data. May not exist for pulse.".format(dtyp)
            )

        DATA.MSTAt = t
        DATA.MSTAx = x
        DATA.EFUS = data
        print(DATA.EFUS)
        """

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
            setattr(DATA, param, data)
        DATA.MAGC_t = t
        DATA.MAGC_x = x

        # may need to interpolate here to sort out different timesteps, not for RNN but y for MLP
        print("timing:")
        print(len(self.MAGC_t))
        print(len(self.EFIT_t))

        return self


# Main function to run whole thing
class Main:
    def __init__(self):
        # pulse_num = 86320
        EFIT_params = ["FAXS", "AREA", "BTPD", "VOLM", "BTND", "BTNM", "BTPD", "XIP"]
        MAGC_params = ["FLX", "IPLA", "BVAC"]
        # XIP automatically extracted
        # params_to_retrieve = input("EFIT params requested:")
        # pulse_num = input("Pulse number:")
        data_thread = DATA(EFIT_params, MAGC_params)

        # Extract multiple pulses
        for pulse_num in range(99070, 99072):
            try:
                data_thread = data_thread.set_pulse(pulse_num)
            except Exception as e:
                print("Data for", pulse_num, "not found. Possibly dry run, skipping.")
                print(e)
                continue
            all_data = {}
            params_to_retrieve = EFIT_params + MAGC_params
            for parameter in params_to_retrieve:
                all_data[parameter] = getattr(data_thread, parameter)
            all_data["EFIT Time"] = DATA.EFIT_t
            all_data["MAGC Time"] = DATA.MAGC_t

            for key, value in all_data.items():
                print(key, len([item for item in value if item]))

            """
            df = pd.DataFrame(all_data)
            df = df.set_index("Time")
            filename = str(pulse_num) + "_EFIT.csv"
            with open(filename, mode="w") as f:
                df.to_csv(f)
            """


#    gtk_thread = gtk_class(data_thread)
#    gtk.main()

Main()
