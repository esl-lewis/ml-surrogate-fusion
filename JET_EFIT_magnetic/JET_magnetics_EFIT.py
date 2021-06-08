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

# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *

# --- data functions
class DATA:
    # Initialise global variables
    def __init__(self, EFIT_params):  # MACHINE DEPENDENT
        # DATA.pulse = 82631
        # DATA.pulse = pulse_number
        DATA.pulse = 0
        DATA.t = np.array([])
        DATA.psi_shift = 0.0
        DATA.params = EFIT_params
        for param in DATA.params:
            setattr(DATA, param, np.array([]))

    # https://stackoverflow.com/questions/32721580/example-of-class-with-user-input
    """
    @classmethod
    def from_input(cls):
        return cls(
            raw_input('Name: '),
            int(raw_input('User ID: ')), 
            int(raw_input('Reputation: ')),
        )
    """

    # --- Load the pulse basic data # MACHINE DEPENDENT
    def set_pulse(self, pulse_number):

        # where EFIT_params is a list of all the metrics we want to retrieve

        # DATA.pulse = int(DATA.pulse_box.get_text())
        DATA.pulse = pulse_number

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)

        # --- Load EFIT data # MACHINE DEPENDENT
        # Load XIP first to sort out t and x
        dda = "EFIT"
        dtyp = "XIP"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise Exception("Failed to load XIP data. May not exist for pulse.")
        DATA.EFITt = t
        DATA.EFITx = x
        DATA.XIP = data

        # need to grab EFIT time once at beginning

        for param in self.params:
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

        # --- Load MAGC data # MACHINE DEPENDENT
        dda = "MSTA"
        dtyp = "EFUS"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            raise IOError(
                "Failed to load {} data. May not exist for pulse.".format(dtyp)
            )

        # Check which magnetics value is usable
        DATA.MSTAt = t
        DATA.MSTAx = x
        DATA.EFUS = data
        print(DATA.EFUS)
        print('test data',data)
        # is this a true/false value?

        if DATA.EFUS == True:
            for param in ["BVAC", "FLX", "IPLA"]:
                dda = "MAGC"
                dtyp = param
                ihdat, iwdat, data, x, t, ier = ppfget(
                    DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
                )
                if ier != 0:
                    raise IOError(
                        "Failed to load {} data. May not exist for pulse.".format(dtyp)
                    )
                setattr(DATA, param, data)
            setattr(DATA, "MAGCx", x)
            setattr(DATA, "MAGCt", t)
            # may need to interpolate here to sort out different timesteps, not for RNN but y for MLP

        print("timing:")
        print(DATA.MSTAt)
        print(DATA.MAGCt)
        print(DATA.EFITt)

        return self


# Main function to run whole thing
class Main:
    def __init__(self):
        # pulse_num = 86320
        params_to_retrieve = ["FAXS", "AREA", "BTPD", "VOLM", "BTND", "BTNM", "BTPD"]
        # XIP automatically extracted
        # params_to_retrieve = input("EFIT params requested:")
        # pulse_num = input("Pulse number:")
        data_thread = DATA(params_to_retrieve)

        # Extract multiple pulses
        for pulse_num in range(99055, 99058):
            try:
                data_thread = data_thread.set_pulse(pulse_num)
            except Exception as error:
                print("Data for", pulse_num, "not found. Probably dry run, skipping.")
                print(error)
                continue
            all_data = {}
            for parameter in params_to_retrieve:
                all_data[parameter] = getattr(data_thread, parameter)
            retrieved_time = DATA.t
            all_data["Time"] = retrieved_time

            plasma_current = DATA.XIP
            all_data["XIP"] = plasma_current

            df = pd.DataFrame(all_data)
            df = df.set_index("Time")
            filename = str(pulse_num) + "_EFIT.csv"
            with open(filename, mode="w") as f:
                df.to_csv(f)


#    gtk_thread = gtk_class(data_thread)
#    gtk.main()

Main()
