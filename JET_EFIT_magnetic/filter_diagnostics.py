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

# TODO grabs MSTA correct probes
# TODO interpolates times

# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *

# --- data functions
class DATA:
    # Initialise global variables
    def __init__(self):  # MACHINE DEPENDENT
        # DATA.pulse = 82631
        # DATA.pulse = pulse_number
        DATA.pulse = 0
        DATA.STBP = np.array([])
        DATA.STFL = np.array([])
        DATA.psi_shift = 0.0
        #DATA.MAGC_params = MAGC_parameters
        #for param in MAGC_parameters:
        #    setattr(DATA, param, np.array([]))

    # --- Load the pulse basic data # MACHINE DEPENDENT
    def set_pulse(self, pulse_number):
        DATA.pulse = pulse_number

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)

        # --- Use MSTA to find which diagnostics are usable
        # STBP = magnetic probe status
        
        dda = "MSTA"
        dtyp = "STBP"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise IOError(
                "Failed to load {} data. May not exist for pulse.".format(dtyp)
            )
        DATA.STBP = data
        print(DATA.STBP)


        # STBP = flux probe status
        dda = "MSTA"
        dtyp = "STFL"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise IOError(
                "Failed to load {} data. May not exist for pulse.".format(dtyp)
            )
        DATA.STFL = data
        print(DATA.STFL)
        print(type(DATA.STFL))


        return self


# Main function to run whole thing
class Main:
    def __init__(self):
        data_thread = DATA()
        STBP_count = np.array([])
        STFL_count = np.array([])
        # Extract multiple pulses
        for pulse_num in range(99070, 99072):
            print('PULSE NUMBER',pulse_num)
            try:
                data_thread = data_thread.set_pulse(pulse_num)
            except Exception as e:
                print("Data for", pulse_num, "not found. Possibly dry run, skipping.")
                print(e)
                continue
            all_data = {}

            for parameter in ['STBP','STFL']:
                all_data[parameter] = getattr(data_thread, parameter)

            for key, value in all_data.items():
                print(key, len([item for item in value if item]))

            # TODO rank most robust diagnostics here

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
