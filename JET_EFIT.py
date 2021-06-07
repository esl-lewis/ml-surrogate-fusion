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

#pulse_number = input("Pulse number:")
pulse_number = 82630

# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *

# --- data functions
class DATA:
    # Initialise global variables
    def __init__(self, EFIT_params):  # MACHINE DEPENDENT
        DATA.pulse = 82631
        # DATA.pulse = pulse_number
        DATA.t = np.array([])
        DATA.t_min = 60.5
        DATA.t_max = 62.0
        DATA.psi_shift = 0.0
        DATA.params = EFIT_params
        for param in DATA.params:
            print("I'm working")
            print(param)
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
    def set_pulse(self):

        # where EFIT_params is a list of all the metrics we want to retrieve

        # DATA.pulse = int(DATA.pulse_box.get_text())
        DATA.pulse = 82630

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)

        # --- Load EFIT data # MACHINE DEPENDENT
        # Load XIP first to sort out t and x
        dda = "EFIT"
        dtyp = "XIP"
        try:
            ihdat, iwdat, data, x, t, ier = ppfget(
                DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
            )
        except:
            print("Failed to load EFIT data, pulse number may not exist")
            return
        DATA.t = t
        DATA.x = x
        DATA.XIP = data

        # need to grab EFIT time once at beginning

        for param in self.params:
            dtyp = str(param)
            ihdat, iwdat, data, x, t, ier = ppfget(
                DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
            )
            if ier != 0:
                print("Failed to load {} data. May not exist for pulse.".format(dtyp))
                return
            # DATA.EFIT_xip = data
            setattr(DATA, param, data)
        print(DATA.AREA[0:4])
        print(DATA.XIP[0:5])
        return self


# Main function to run whole thing
class Main:
    def __init__(self):
        params_to_retrieve = ["AREA", "BTPD"]
        # params_to_retrieve = input("EFIT params requested:")
        data_thread = DATA(params_to_retrieve)
        data_thread = data_thread.set_pulse()
        all_data = {}
        for parameter in params_to_retrieve:
            all_data[parameter] = getattr(data_thread,parameter)
        retrieved_time = DATA.t
        all_data['Time'] = retrieved_time
        print(data_thread.AREA[0:6])
        print(dir(data_thread))
        #print(all_data)
        
        """
        df = pd.DataFrame(all_data)

        filename = str(pulse_number) + "_EFIT.csv"
        with open(filename, mode="w") as f:
            df.to_csv(f)
        """


#    gtk_thread = gtk_class(data_thread)
#    gtk.main()

Main()
