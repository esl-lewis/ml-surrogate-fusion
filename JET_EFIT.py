from numpy.core.arrayprint import DatetimeFormat


try:
    import sys, traceback
    import numpy
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

pulse_number = input("Pulse number:")


# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *

# --- data functions
class DATA:
    # Initialise global variables
    def __init__(self):  # MACHINE DEPENDENT
        #DATA.pulse = 82631
        DATA.pulse = pulse_number
        DATA.t_min = 60.5
        DATA.t_max = 62.0
        DATA.i_min = 0
        DATA.i_max = 0
        DATA.psi_shift = 0.0
        DATA.x = numpy.array([])
        DATA.y_ne = numpy.array([])
        DATA.y_Te = numpy.array([])
        DATA.y_psi = numpy.array([])
        DATA.ne = numpy.array([])
        DATA.Te = numpy.array([])
        DATA.ne_coef = numpy.zeros(7)
        DATA.Te_coef = numpy.zeros(7)
        DATA.psi = numpy.array([])
        DATA.NBI_t = numpy.array([])
        DATA.NBI_ptot = numpy.array([])

    # --- Load the pulse basic data # MACHINE DEPENDENT
    def set_pulse(self, widget):
        #DATA.pulse = int(DATA.pulse_box.get_text())
        DATA.pulse = pulse_number

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)


        # --- Load EFIT data # MACHINE DEPENDENT
        dda = "EFIT"
        dtyp = "XIP"
        try:
            ihdat, iwdat, data, x, t, ier = ppfget(
                DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
            )
        except:
            print("Failed to load EFIT data, pulse number may not exist")
        DATA.EFIT_t = t
        DATA.EFIT_xip = data
        return(data,t)


# Main function to run whole thing
class Main:
    def __init__(self):
        data_thread = DATA()
        retrieved_data, retrieved_time = data_thread.set_pulse(data_thread)
        print(len(retrieved_data))
        print(len(retrieved_time))
        print(retrieved_data[0:4])
        print(retrieved_time[0:4])
        data_dict = {'efit':retrieved_data,'time':retrieved_time}
        df = pd.DataFrame(data_dict)
        
        filename = str(pulse_number) + '_EFIT.csv'
        with open(filename,mode='w') as f:
            df.to_csv(f)


#    gtk_thread = gtk_class(data_thread)
#    gtk.main()

Main()
