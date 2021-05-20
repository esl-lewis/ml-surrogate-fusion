from numpy.core.arrayprint import DatetimeFormat


try:
    import sys
    import sys, traceback
    import numpy
    import pylab
    import pyuda
    import py_flush as Flush
    from scipy import optimize

    # from math import pi, sin, cos, sqrt, exp, atan2, tanh, cosh
    import gtk
    import numpy
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

# --- Modules for JETPPF system
sys.path[:0] = ["/jet/share/lib/python"]
from ppf import *

# including ppfget I assume

# import py_flush as Flush


"""
get rid of classes
get rid of gtk is just GUI

care about reading functions, load efit data, ppf get is what we care about 

remove everything below Data

we only care about magnetics, not pressure + other physical stuff
"""


# --- data functions
class DATA:
    # I--- nitialise global variables
    def __init__(self):  # MACHINE DEPENDENT
        DATA.pulse = 82630
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
        DATA.pulse = int(DATA.pulse_box.get_text())

        # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
        ier = ppfgo(pulse=DATA.pulse, seq=0)
        ppfuid("JETPPF", rw="R")
        ier = ppfgo(pulse=DATA.pulse, seq=0)

        # --- Load HRTS data # MACHINE DEPENDENT
        dda = "HRTS"
        dtyp = "NE"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            self.pop_up_message(self, "failed to load HRTS/NE data")
            return
        DATA.t = t
        DATA.x = x
        DATA.y_ne = data
        dtyp = "TE"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            self.pop_up_message(self, "failed to load HRTS/TE data")
            return
        DATA.y_Te = data
        dtyp = "PSI"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            self.pop_up_message(self, "failed to load HRTS/PSI data")
            return
        DATA.y_psi = data

        # --- Load EFIT data # MACHINE DEPENDENT
        dda = "EFIT"
        dtyp = "XIP"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            self.pop_up_message(self, "failed to load EFIT/XIP data")
            return
        DATA.EFIT_t = t
        DATA.EFIT_xip = data
        dda = "NBI"
        dtyp = "PTOT"
        ihdat, iwdat, data, x, t, ier = ppfget(
            DATA.pulse, dda, dtyp, fix0=0, reshape=0, no_x=0, no_t=0
        )
        if ier != 0:
            self.pop_up_message(self, "failed to load NBI/PTOT data")
            return
        DATA.NBI_t = t
        DATA.NBI_ptot = data

    # --- Set the time interval and select ne-Te profiles in this interval # MACHINE DEPENDENT
    def set_time(self, widget):
        DATA.t_min = float(DATA.t_min_box.get_text())
        DATA.t_max = float(DATA.t_max_box.get_text())
        DATA.psi_shift = float(DATA.shift_box.get_text())

        # --- check times
        if DATA.t_max <= DATA.t_min:
            self.pop_up_message(self, "t-max needs to be\nsmaller than t-min")
            return

        # --- Find min/max times in our data set
        nt = numpy.size(DATA.t)
        nx = numpy.size(DATA.x)
        tmin_tmp = +1.0e10
        tmax_tmp = +1.0e10
        imin_tmp = 0
        imax_tmp = 0
        for i in range(nt):
            if abs(DATA.t[i] - DATA.t_min) < tmin_tmp:
                tmin_tmp = abs(DATA.t[i] - DATA.t_min)
                imin_tmp = i
        for i in range(nt):
            if abs(DATA.t[i] - DATA.t_max) < tmax_tmp:
                tmax_tmp = abs(DATA.t[i] - DATA.t_max)
                imax_tmp = i
        DATA.i_min = imin_tmp
        DATA.i_max = imax_tmp

        # --- Just make big arrays with all the data points # MACHINE DEPENDENT (this depends on how your (t,x,y) data is structured)
        DATA.ne = DATA.y_ne[DATA.i_min * nx : DATA.i_max * nx]
        DATA.Te = DATA.y_Te[DATA.i_min * nx : DATA.i_max * nx]
        DATA.psi = DATA.y_psi[DATA.i_min * nx : DATA.i_max * nx] + DATA.psi_shift

    # --- Pop-up window for error messages
    def pop_up_message(self, widget, message):
        self.pop_window = gtk.Window()
        self.pop_window.set_position(gtk.WIN_POS_CENTER)
        self.pop_window.set_size_request(400, 400)
        self.pop_window.set_title("Warning!")
        self.VB1 = gtk.VBox()
        self.pop_window.add(self.VB1)
        self.VB1.add(gtk.Label(message))
        self.close_button = gtk.Button("ok")
        self.close_button.connect("clicked", self.close_pop_up)
        self.VB1.add(self.close_button)
        self.pop_window.show_all()
        return

    # --- Close pop-up window
    def close_pop_up(self, widget):
        self.pop_window.destroy()

