#!/usr/bin/env python


# execute code as script directly in terminal, after "chmod +x"
# GUI should be self-explanatory, please report bugs and comments to S.Pamela
# Code to fit ne-TE profiles with JOREK fit-functions
# Author: S.Pamela
# Contact: stanislas.pamela@ukaea.uk
# Date: March-2019




#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#-------------------------------------- MODULES -------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


# --- basic python modules
import numpy
import sys
from scipy import optimize

# --- Modules for gtk and matplotlib together
import gtk
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.figure import Axes
from matplotlib.backends.backend_gtkagg import (
    NavigationToolbar2GTK as NavigationToolbar)


# --- Global variables
mu_0	     = 4*3.1415926535*1e-7
eV2Joules    = 1.602176487*1e-19
scale_height = 50

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#-------------------------------------- DATA CLASS (MACHINE DEPENDENT) --------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


# --- Modules for JETPPF system
sys.path[:0]=['/jet/share/lib/python']
from ppf import *
#import py_flush as Flush

# --- data functions
class DATA():
  # I--- nitialise global variables
  def __init__(self): # MACHINE DEPENDENT
    DATA.pulse     = 82630
    DATA.t_min     = 60.5
    DATA.t_max     = 62.0
    DATA.i_min     = 0
    DATA.i_max     = 0
    DATA.psi_shift = 0.0
    DATA.x         = numpy.array([])
    DATA.y_ne      = numpy.array([])
    DATA.y_Te      = numpy.array([])
    DATA.y_psi     = numpy.array([])
    DATA.ne        = numpy.array([])
    DATA.Te        = numpy.array([])
    DATA.ne_coef   = numpy.zeros(7)
    DATA.Te_coef   = numpy.zeros(7)
    DATA.psi       = numpy.array([])
    DATA.NBI_t     = numpy.array([])
    DATA.NBI_ptot  = numpy.array([])
    
  # --- Load the pulse basic data # MACHINE DEPENDENT
  def set_pulse(self, widget):
    DATA.pulse = int(DATA.pulse_box.get_text())
    
    # --- Prepare JETPPF system (yes, you need to knock before you open it...) # MACHINE DEPENDENT
    ier = ppfgo(pulse=DATA.pulse, seq=0)
    ppfuid("JETPPF",rw="R")
    ier = ppfgo(pulse=DATA.pulse,seq=0)
    
    # --- Load HRTS data # MACHINE DEPENDENT
    dda = "HRTS"
    dtyp = "NE"
    ihdat,iwdat,data,x,t,ier = ppfget(DATA.pulse,dda,dtyp,fix0=0,reshape=0,no_x=0,no_t=0)
    if (ier != 0):
      self.pop_up_message(self,"failed to load HRTS/NE data")
      return
    DATA.t = t
    DATA.x = x
    DATA.y_ne = data
    dtyp = "TE"
    ihdat,iwdat,data,x,t,ier = ppfget(DATA.pulse,dda,dtyp,fix0=0,reshape=0,no_x=0,no_t=0)
    if (ier != 0):
      self.pop_up_message(self,"failed to load HRTS/TE data")
      return
    DATA.y_Te = data
    dtyp = "PSI"
    ihdat,iwdat,data,x,t,ier = ppfget(DATA.pulse,dda,dtyp,fix0=0,reshape=0,no_x=0,no_t=0)
    if (ier != 0):
      self.pop_up_message(self,"failed to load HRTS/PSI data")
      return
    DATA.y_psi = data
    
    # --- Load EFIT data # MACHINE DEPENDENT
    dda = "EFIT"
    dtyp = "XIP"
    ihdat,iwdat,data,x,t,ier = ppfget(DATA.pulse,dda,dtyp,fix0=0,reshape=0,no_x=0,no_t=0)
    if (ier != 0):
      self.pop_up_message(self,"failed to load EFIT/XIP data")
      return
    DATA.EFIT_t   = t
    DATA.EFIT_xip = data
    dda = "NBI"
    dtyp = "PTOT"
    ihdat,iwdat,data,x,t,ier = ppfget(DATA.pulse,dda,dtyp,fix0=0,reshape=0,no_x=0,no_t=0)
    if (ier != 0):
      self.pop_up_message(self,"failed to load NBI/PTOT data")
      return
    DATA.NBI_t    = t
    DATA.NBI_ptot = data
  
  # --- Set the time interval and select ne-Te profiles in this interval # MACHINE DEPENDENT
  def set_time(self, widget):
    DATA.t_min     = float(DATA.t_min_box.get_text())
    DATA.t_max     = float(DATA.t_max_box.get_text())
    DATA.psi_shift = float(DATA.shift_box.get_text())
    
    # --- check times
    if (DATA.t_max <= DATA.t_min):
      self.pop_up_message(self,"t-max needs to be\nsmaller than t-min")
      return
    
    # --- Find min/max times in our data set
    nt = numpy.size(DATA.t)
    nx = numpy.size(DATA.x)
    tmin_tmp = +1.e10
    tmax_tmp = +1.e10
    imin_tmp = 0
    imax_tmp = 0
    for i in range(nt):
      if (abs(DATA.t[i]-DATA.t_min) < tmin_tmp):
        tmin_tmp = abs(DATA.t[i]-DATA.t_min)
        imin_tmp = i
    for i in range(nt):
      if (abs(DATA.t[i]-DATA.t_max) < tmax_tmp):
        tmax_tmp = abs(DATA.t[i]-DATA.t_max)
        imax_tmp = i
    DATA.i_min = imin_tmp
    DATA.i_max = imax_tmp
    
    # --- Just make big arrays with all the data points # MACHINE DEPENDENT (this depends on how your (t,x,y) data is structured)
    DATA.ne  = DATA.y_ne [DATA.i_min*nx:DATA.i_max*nx]
    DATA.Te  = DATA.y_Te [DATA.i_min*nx:DATA.i_max*nx]
    DATA.psi = DATA.y_psi[DATA.i_min*nx:DATA.i_max*nx] + DATA.psi_shift
  
  # --- Pop-up window for error messages
  def pop_up_message(self,widget,message):
    self.pop_window = gtk.Window()
    self.pop_window.set_position(gtk.WIN_POS_CENTER)
    self.pop_window.set_size_request(400, 400)
    self.pop_window.set_title ("Warning!")
    self.VB1 = gtk.VBox()
    self.pop_window.add(self.VB1)
    self.VB1.add(gtk.Label(message))
    self.close_button = gtk.Button("ok")
    self.close_button.connect("clicked", self.close_pop_up)
    self.VB1.add(self.close_button)
    self.pop_window.show_all()
    return
  
  # --- Close pop-up window
  def close_pop_up(self,widget):
    self.pop_window.destroy()




#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------- JOREK Fitting function -----------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def residual(a, y, x):
    err = y-curve(x,a)
    return err


def curve(x, a):
    poly  = 1.0 + a[2]*x + a[3]*x*x + a[4]*x*x*x
    atanh = 0.5 - 0.5*numpy.tanh((x-a[6])/a[5])
    return (a[0]-a[1]) * poly * atanh + a[1]


def fit_data(x_input, y_input):
    xx = numpy.array([])
    yy = numpy.array([])
    for i in range(numpy.size(DATA.ne)):
      xx = numpy.append(xx,x_input[i])
      yy = numpy.append(yy,y_input[i])
    a0  = [0.8*max(y_input), 0.0, 0.0, 0.0, 0.0, 0.03, 0.9] # Initial guess for the parameters
    a1  = optimize.leastsq(residual, a0, args=(yy,xx), maxfev=2000)
    a1 = a1[0]
    x_fit = numpy.linspace(min(x_input),max(x_input),200)
    y_fit = numpy.array([])
    for i in range (0,numpy.size(x_fit)):
        y_fit = numpy.append(y_fit,curve(x_fit[i],a1))
    return a1, x_fit, y_fit


def get_fitted_curve(a1,xmin,xmax):
    x_fit = numpy.linspace(xmin,xmax,200)
    y_fit = numpy.array([])
    for i in range (0,numpy.size(x_fit)):
        y_fit = numpy.append(y_fit,curve(x_fit[i],a1))
    return x_fit, y_fit




#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#-------------------------------------- GTK CLASS (HANDLES INTERACTIVE WINDOWS) -----------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------




# --- gtk functions
class gtk_class(DATA):
  # --------------------------
  # --- Initialise application
  def __init__(self, data_thread):
    gtk_class.window_partition = -1
    
    # --- Setup the window and the layout manager
    self.window = gtk.Window()
    self.window.connect("destroy", gtk.main_quit)
    self.window.set_title ("JOREK fit-script for JET-HRTS profiles")
    #self.window.maximize()
    #self.window.set_size_request(800, 800)
    
    # --- This is the box that contains everything
    self.VB1 = gtk.VBox()
    self.HB1 = gtk.HBox()
    self.VB1.add(self.HB1)
    self.window.add(self.VB1)

    # --- Entry for the pulse number
    self.HB1.add(gtk.Label("pulse number:"))
    DATA.pulse_box = gtk.Entry()
    DATA.pulse_box.set_text("82630")
    self.HB1.add(DATA.pulse_box)

    # --- Button to load pulse data
    self.load_button = gtk.Button("load")
    self.load_button.connect("clicked", data_thread.set_pulse)
    self.load_button.connect("clicked", self.plot_pulse)
    self.HB1.add(self.load_button)

    # --- Entry for the min time
    self.HB1.add(gtk.Label("t-min:"))
    DATA.t_min_box = gtk.Entry()
    DATA.t_min_box.set_text(str(DATA.t_min))
    self.HB1.add(DATA.t_min_box)

    # --- Entry for the end time
    self.HB1.add(gtk.Label("t-max:"))
    DATA.t_max_box = gtk.Entry()
    DATA.t_max_box.set_text(str(DATA.t_max))
    self.HB1.add(DATA.t_max_box)

    # --- Entry for the psi shift
    self.HB1.add(gtk.Label("psi-shift:"))
    DATA.shift_box = gtk.Entry()
    DATA.shift_box.set_text(str(DATA.psi_shift))
    self.HB1.add(DATA.shift_box)

    # --- Button to load pulse data
    self.load_button = gtk.Button("set time & shift")
    self.load_button.connect("clicked", data_thread.set_time)
    self.load_button.connect("clicked", self.plot_time)
    self.HB1.add(self.load_button)

    # --- Button for test-plot (for now...)
    self.plot_button = gtk.Button("fit ne")
    self.plot_button.connect("clicked", data_thread.set_time)
    self.plot_button.connect("clicked", self.plot_ne)
    self.HB1.add(self.plot_button)

    # --- Button for test-plot (for now...)
    self.plot_button = gtk.Button("fit Te")
    self.plot_button.connect("clicked", data_thread.set_time)
    self.plot_button.connect("clicked", self.plot_Te)
    self.HB1.add(self.plot_button)

    self.plot_empty()
    data_thread.set_pulse(data_thread)
    self.plot_pulse(self)
    
    # --- Show the application
    self.window.show_all()

  # ----------------------------------------
  # --- Plots for the pulse
  # --- Start with an empty plot as initialisation
  def plot_empty(self):
    # --- The plot
    self.figure = Figure()
    self.ax = self.figure.add_subplot(111)
    x_plot = numpy.linspace(0.0,1.0,100)
    y_plot = x_plot * 0.0
    self.ax.plot(x_plot,y_plot,lw=2)
    self.canvas = FigureCanvas(self.figure)
    self.canvas.set_size_request(400, 400)
    self.HB2 = gtk.HBox()
    self.VB2 = gtk.VBox()
    self.HB2.add(self.VB2)
    self.VB1.add(self.HB2)
    self.VB2.pack_start(self.canvas, True, True, 0)
    toolbar = NavigationToolbar(self.canvas, self.window)
    self.VB2.pack_start(toolbar, False, False, 0)
    # --- The initial coefs (just random...)
    self.VB3 = gtk.VBox()
    self.rho_coefs =                  " ! --- rho_coefs:\n"
    self.rho_coefs = self.rho_coefs + " central_density = "+str(0.5 )+"\n"
    self.rho_coefs = self.rho_coefs + " rho_0           = "+str(1.0 )+"\n"
    self.rho_coefs = self.rho_coefs + " rho_1           = "+str(0.01)+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(1)     = "+str(-1.0)+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(2)     = "+str(0.0 )+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(3)     = "+str(0.0 )+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(4)     = "+str(0.03)+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(5)     = "+str(0.98)+"\n"
    self.rho_coefs_print = gtk.Label(self.rho_coefs)
    self.rho_coefs_print.set_selectable(True)
    self.rho_coefs_print.set_justify(gtk.JUSTIFY_LEFT)
    self.VB3.add(self.rho_coefs_print)
    self.T_coefs =                " ! --- T_coefs:\n"
    self.T_coefs = self.T_coefs + " ! --- central Te temperature: " + str(1.0) + "eV\n"
    self.T_coefs = self.T_coefs + " T_0           = "+str(1.0 )+"\n"
    self.T_coefs = self.T_coefs + " T_1           = "+str(0.01)+"\n"
    self.T_coefs = self.T_coefs + " T_coef(1)     = "+str(-1.0)+"\n"
    self.T_coefs = self.T_coefs + " T_coef(2)     = "+str(0.0 )+"\n"
    self.T_coefs = self.T_coefs + " T_coef(3)     = "+str(0.0 )+"\n"
    self.T_coefs = self.T_coefs + " T_coef(4)     = "+str(0.03)+"\n"
    self.T_coefs = self.T_coefs + " T_coef(5)     = "+str(0.98)+"\n"
    self.T_coefs_print = gtk.Label(self.T_coefs)
    self.T_coefs_print.set_selectable(True)
    self.T_coefs_print.set_justify(gtk.JUSTIFY_LEFT)
    self.VB3.add(self.T_coefs_print)
    self.VB3.set_property("width-request", 10)
    self.HB2.add(self.VB3)
    self.window.show_all()

  # --- Plot basic pulse data
  def plot_pulse(self, data_thread):
    self.figure.clf()
    self.ax = self.figure.add_subplot(111)
    self.ax.plot(DATA.EFIT_t,abs(DATA.EFIT_xip*1.e-6),'k-')
    self.ax.plot(DATA.NBI_t,DATA.NBI_ptot*1.e-6,'r-')
    self.ax.legend(['Ip [MA]','NBI [MW]'])
    self.ax.set_xlabel('time [s]')
    self.ax.set_title('JET pulse #' + str(DATA.pulse))
    xx = [DATA.t_min, DATA.t_min]
    y_max = self.ax.get_ylim()
    y_max = y_max[1]
    yy = [0.0,y_max]
    self.ax.plot(xx,yy,'k--')
    xx = [DATA.t_max, DATA.t_max]
    self.ax.plot(xx,yy,'k--')
    self.window.show_all()
    self.plots_update()
    
  
  # --- Plot lines for the time
  def plot_time(self, data_thread):
    self.plot_pulse(self) # just to remove previous lines...
    xx = [DATA.t_min, DATA.t_min]
    y_max = self.ax.get_ylim()
    y_max = y_max[1]
    yy = [0.0,y_max]
    self.ax.plot(xx,yy,'k--')
    xx = [DATA.t_max, DATA.t_max]
    self.ax.plot(xx,yy,'k--')
    self.window.show_all()
    #self.plots_update()
    
  # --- Need resize to update gtk???
  def plots_update(self):
    gtk_class.window_partition = -gtk_class.window_partition
    xx = self.HB1.allocation.width
    yy = self.HB1.allocation.height + gtk_class.window_partition
    self.HB1.set_size_request(xx,yy)
    xx = self.HB2.allocation.width
    yy = self.HB2.allocation.height - gtk_class.window_partition
    self.HB2.set_size_request(xx,yy)
  
  # ----------------------------------------
  # --- Plots for ne fit
  # --- Fit ne data and plot
  def plot_ne(self, data_thread):
    # --- Open new window
    self.ne_window = gtk.Window()
    self.ne_window.set_position(gtk.WIN_POS_CENTER)
    self.ne_window.set_property("width-request", 1000)
    self.ne_window.set_property("height-request", 600)
    self.ne_window.set_title ("ne fit")
    self.ne_HB = gtk.HBox()
    self.ne_VB = gtk.VBox()
    self.ne_HB.add(self.ne_VB)
    self.ne_window.add(self.ne_HB)
    
    # --- Open new figure canvas with ne data
    self.ne_figure = Figure()
    self.ne_ax = self.ne_figure.add_subplot(111)
    self.ne_ax.plot(DATA.psi,DATA.ne,'ko')
    
    # --- Fit density data and plot fit
    DATA.ne_coef, psi_fit, ne_fit  = fit_data(DATA.psi, DATA.ne)
    self.line_rho = self.ne_ax.plot(psi_fit,ne_fit,'b',lw=2)
    
    # --- Show title of adjustment (IMPORTANT, used together with ne_B1 to resize=update plot)
    self.ne_B2 = gtk.HBox()
    self.ne_B2.set_property("height-request", 20)
    self.ne_B2.add(gtk.Label("adjust JOREK profile:"))
    self.ne_VB.add(self.ne_B2)
    
    # --- Show plot with tool-bar on canvas
    self.ne_canvas = FigureCanvas(self.ne_figure)
    self.ne_B1 = gtk.VBox()
    self.ne_B1.set_property("width-request", 400)
    self.ne_B1.set_property("height-request", 400)
    self.ne_VB.add(self.ne_B1)
    self.ne_B1.pack_start(self.ne_canvas, True, True, 0)
    toolbar = NavigationToolbar(self.ne_canvas, self.ne_window)
    self.ne_B1.pack_start(toolbar, False, False, 0)
    
    # --- Button to save and close
    self.close_ne = gtk.Button("close")
    self.close_ne.connect("clicked", self.close_ne_window)
    self.ne_VB.add(self.close_ne)
    
    # --- Box for the adjustment bars
    self.ne_VB2 = gtk.VBox()
    self.ne_HB.add(self.ne_VB2)
    self.scale_rho_0 = gtk.HBox()
    self.scale_rho_0.set_property("height-request", scale_height)
    self.scale_rho_0.add(gtk.Label("central_density [1.e19m-3]"))
    val_init = DATA.ne_coef[0]*1.e-19
    val_min  = 0.2*max(DATA.ne)*1.e-19
    val_max  = 2.0*max(DATA.ne)*1.e-19
    adj_rho_0 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=1, page_incr=10, page_size=0) 
    adj_rho_0.connect( "value_changed", self.adjust_rho_0)
    adj_rho_0.emit( "value_changed")
    self.h_scale_rho_0 = gtk.HScale(adjustment=adj_rho_0)
    self.h_scale_rho_0.set_digits(1)
    self.scale_rho_0.add(self.h_scale_rho_0)
    self.ne_VB2.add(self.scale_rho_0)
    
    # --- Box for the adjustment bars
    self.scale_rho_1 = gtk.HBox()
    self.scale_rho_1.set_property("height-request", scale_height)
    self.scale_rho_1.add(gtk.Label("rho_1 (SOL-value)"))
    val_init = DATA.ne_coef[1] / DATA.ne_coef[0]
    val_min  = 0.0
    val_max  = 0.3
    adj_rho_1 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_rho_1.connect( "value_changed", self.adjust_rho_1)
    adj_rho_1.emit( "value_changed")
    self.h_scale_rho_1 = gtk.HScale(adjustment=adj_rho_1)
    self.h_scale_rho_1.set_digits(2)
    self.scale_rho_1.add(self.h_scale_rho_1)
    self.ne_VB2.add(self.scale_rho_1)
    
    # --- Box for the adjustment bars
    self.scale_rho_C1 = gtk.HBox()
    self.scale_rho_C1.set_property("height-request", scale_height)
    self.scale_rho_C1.add(gtk.Label("rho_coef(1) (slope)"))
    val_init = DATA.ne_coef[2]
    val_min  = -3.0
    val_max  = +3.0
    adj_rho_C1 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_rho_C1.connect( "value_changed", self.adjust_rho_C1)
    adj_rho_C1.emit( "value_changed")
    self.h_scale_rho_C1 = gtk.HScale(adjustment=adj_rho_C1)
    self.h_scale_rho_C1.set_digits(2)
    self.scale_rho_C1.add(self.h_scale_rho_C1)
    self.ne_VB2.add(self.scale_rho_C1)
    
    # --- Box for the adjustment bars
    self.scale_rho_C2 = gtk.HBox()
    self.scale_rho_C2.set_property("height-request", scale_height)
    self.scale_rho_C2.add(gtk.Label("rho_coef(2) (slope^2)"))
    val_init = DATA.ne_coef[3]
    val_min  = -3.0
    val_max  = +3.0
    adj_rho_C2 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_rho_C2.connect( "value_changed", self.adjust_rho_C2)
    adj_rho_C2.emit( "value_changed")
    self.h_scale_rho_C2 = gtk.HScale(adjustment=adj_rho_C2)
    self.h_scale_rho_C2.set_digits(2)
    self.scale_rho_C2.add(self.h_scale_rho_C2)
    self.ne_VB2.add(self.scale_rho_C2)
    
    # --- Box for the adjustment bars
    self.scale_rho_C3 = gtk.HBox()
    self.scale_rho_C3.set_property("height-request", scale_height)
    self.scale_rho_C3.add(gtk.Label("rho_coef(3) (slope^3)"))
    val_init = DATA.ne_coef[4]
    val_min  = -3.0
    val_max  = +3.0
    adj_rho_C3 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_rho_C3.connect( "value_changed", self.adjust_rho_C3)
    adj_rho_C3.emit( "value_changed")
    self.h_scale_rho_C3 = gtk.HScale(adjustment=adj_rho_C3)
    self.h_scale_rho_C3.set_digits(2)
    self.scale_rho_C3.add(self.h_scale_rho_C3)
    self.ne_VB2.add(self.scale_rho_C3)
    
    # --- Box for the adjustment bars
    self.scale_rho_C4 = gtk.HBox()
    self.scale_rho_C4.set_property("height-request", scale_height)
    self.scale_rho_C4.add(gtk.Label("rho_coef(4) (ped-width)"))
    val_init = DATA.ne_coef[5]
    val_min  = 0.001
    val_max  = 0.3
    adj_rho_C4 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_rho_C4.connect( "value_changed", self.adjust_rho_C4)
    adj_rho_C4.emit( "value_changed")
    self.h_scale_rho_C4 = gtk.HScale(adjustment=adj_rho_C4)
    self.h_scale_rho_C4.set_digits(3)
    self.scale_rho_C4.add(self.h_scale_rho_C4)
    self.ne_VB2.add(self.scale_rho_C4)
    
    # --- Box for the adjustment bars
    self.scale_rho_C5 = gtk.HBox()
    self.scale_rho_C5.set_property("height-request", scale_height)
    self.scale_rho_C5.add(gtk.Label("rho_coef(5) (ped-pos)"))
    val_init = DATA.ne_coef[6]
    val_min  = 0.6
    val_max  = 1.5
    adj_rho_C5 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_rho_C5.connect( "value_changed", self.adjust_rho_C5)
    adj_rho_C5.emit( "value_changed")
    self.h_scale_rho_C5 = gtk.HScale(adjustment=adj_rho_C5)
    self.h_scale_rho_C5.set_digits(2)
    self.scale_rho_C5.add(self.h_scale_rho_C5)
    self.ne_VB2.add(self.scale_rho_C5)
    
    self.ne_window.show_all()
    
  # --- Adjust fit with scales
  def adjust_rho_0 (self, adj): DATA.ne_coef[0] = adj.value * 1.e19           ; self.update_ne_plot(self)
  def adjust_rho_1 (self, adj): DATA.ne_coef[1] = adj.value * DATA.ne_coef[0] ; self.update_ne_plot(self)
  def adjust_rho_C1(self, adj): DATA.ne_coef[2] = adj.value                   ; self.update_ne_plot(self)
  def adjust_rho_C2(self, adj): DATA.ne_coef[3] = adj.value                   ; self.update_ne_plot(self)
  def adjust_rho_C3(self, adj): DATA.ne_coef[4] = adj.value                   ; self.update_ne_plot(self)
  def adjust_rho_C4(self, adj): DATA.ne_coef[5] = adj.value                   ; self.update_ne_plot(self)
  def adjust_rho_C5(self, adj): DATA.ne_coef[6] = adj.value                   ; self.update_ne_plot(self)
       
  # --- Update ne plot
  def update_ne_plot(self, data_thread):
    psi_fit, ne_fit  = get_fitted_curve(DATA.ne_coef,min(DATA.psi),max(DATA.psi))
    self.line_rho.pop(0).remove()
    self.line_rho = self.ne_ax.plot(psi_fit,ne_fit,'b',lw=2)
    self.ne_ax.set_title('density for JET pulse #' + str(DATA.pulse) + '\nfrom ' + str(DATA.t_min) + 's to ' + str(DATA.t_max) + 's')
    self.ne_ax.set_xlabel('psi [norm]')
    self.ne_ax.set_ylabel('ne [m-3]')
    self.plots_update_ne()
    self.ne_window.show_all()
    self.update_coefs_print(self)
    
  # --- Need resize to update gtk???
  def plots_update_ne(self):
    gtk_class.window_partition = -gtk_class.window_partition
    xx, yy = self.ne_B1.size_request()
    yy = yy + gtk_class.window_partition
    self.ne_B1.set_size_request(xx,yy)
    xx, yy = self.ne_B2.size_request()
    yy = yy - gtk_class.window_partition
    self.ne_B2.set_size_request(xx,yy)

  # --- Close ne-plot and save ne-coefs
  def close_ne_window(self,widget):
    self.ne_window.destroy()
  
  
  
  
  # ----------------------------------------
  # --- Plots for Te fit
  # --- Fit Te data and plot
  def plot_Te(self, data_thread):
    # --- Open new window
    self.Te_window = gtk.Window()
    self.Te_window.set_position(gtk.WIN_POS_CENTER)
    self.Te_window.set_property("width-request", 1000)
    self.Te_window.set_property("height-request", 600)
    self.Te_window.set_title ("Te fit")
    self.Te_HB = gtk.HBox()
    self.Te_VB = gtk.VBox()
    self.Te_HB.add(self.Te_VB)
    self.Te_window.add(self.Te_HB)
    
    # --- Open new figure canvas with ne data
    self.Te_figure = Figure()
    self.Te_ax = self.Te_figure.add_subplot(111)
    self.Te_ax.plot(DATA.psi,DATA.Te,'ko')
    
    # --- Fit density data and plot fit
    DATA.Te_coef, psi_fit, Te_fit  = fit_data(DATA.psi, DATA.Te)
    self.line_T = self.Te_ax.plot(psi_fit,Te_fit,'r',lw=2)
    
    # --- Show title of adjustment (IMPORTANT, used together with ne_B1 to resize=update plot)
    self.Te_B2 = gtk.HBox()
    self.Te_B2.set_property("height-request", 20)
    self.Te_B2.add(gtk.Label("adjust profile:"))
    self.Te_VB.add(self.Te_B2)
    
    # --- Show plot with tool-bar on canvas
    self.Te_canvas = FigureCanvas(self.Te_figure)
    self.Te_B1 = gtk.VBox()
    self.Te_B1.set_property("width-request", 400)
    self.Te_B1.set_property("height-request", 400)
    self.Te_VB.add(self.Te_B1)
    self.Te_B1.pack_start(self.Te_canvas, True, True, 0)
    toolbar = NavigationToolbar(self.Te_canvas, self.Te_window)
    self.Te_B1.pack_start(toolbar, False, False, 0)
    
    # --- Button to save and close
    self.close_Te = gtk.Button("close")
    self.close_Te.connect("clicked", self.close_Te_window)
    self.Te_VB.add(self.close_Te)
    
    # --- Box for the adjustment bars
    self.Te_VB2 = gtk.VBox()
    self.Te_HB.add(self.Te_VB2)
    self.scale_T_0 = gtk.HBox()
    self.scale_T_0.set_property("height-request",scale_height)
    self.scale_T_0.add(gtk.Label("T_0 (axis temperature)"))
    val_init = 2.0 * DATA.Te_coef[0]*DATA.ne_coef[0]*mu_0*eV2Joules
    val_min  = 0.1 * val_init
    val_max  = 3.0 * val_init
    adj_T_0 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=1, page_incr=10, page_size=0) 
    adj_T_0.connect( "value_changed", self.adjust_T_0)
    adj_T_0.emit( "value_changed")
    self.h_scale_T_0 = gtk.HScale(adjustment=adj_T_0)
    self.h_scale_T_0.set_digits(4)
    self.scale_T_0.add(self.h_scale_T_0)
    self.Te_VB2.add(self.scale_T_0)
    
    # --- Box for the adjustment bars
    self.scale_T_1 = gtk.HBox()
    self.scale_T_1.set_property("height-request", scale_height)
    self.scale_T_1.add(gtk.Label("T_1 (SOL-value)"))
    val_init = 2.0   * DATA.Te_coef[1]*DATA.ne_coef[0]*mu_0*eV2Joules
    val_min  = 0.002 * DATA.Te_coef[0]*DATA.ne_coef[0]*mu_0*eV2Joules
    val_max  = 0.3   * DATA.Te_coef[0]*DATA.ne_coef[0]*mu_0*eV2Joules
    adj_T_1 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_T_1.connect( "value_changed", self.adjust_T_1)
    adj_T_1.emit( "value_changed")
    self.h_scale_T_1 = gtk.HScale(adjustment=adj_T_1)
    self.h_scale_T_1.set_digits(5)
    self.scale_T_1.add(self.h_scale_T_1)
    self.Te_VB2.add(self.scale_T_1)
    
    # --- Box for the adjustment bars
    self.scale_T_C1 = gtk.HBox()
    self.scale_T_C1.set_property("height-request", scale_height)
    self.scale_T_C1.add(gtk.Label("T_coef(1) (slope)"))
    val_init = DATA.Te_coef[2]
    val_min  = -3.0
    val_max  = +3.0
    adj_T_C1 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_T_C1.connect( "value_changed", self.adjust_T_C1)
    adj_T_C1.emit( "value_changed")
    self.h_scale_T_C1 = gtk.HScale(adjustment=adj_T_C1)
    self.h_scale_T_C1.set_digits(2)
    self.scale_T_C1.add(self.h_scale_T_C1)
    self.Te_VB2.add(self.scale_T_C1)
    
    # --- Box for the adjustment bars
    self.scale_T_C2 = gtk.HBox()
    self.scale_T_C2.set_property("height-request", scale_height)
    self.scale_T_C2.add(gtk.Label("T_coef(2) (slope^2)"))
    val_init = DATA.Te_coef[3]
    val_min  = -3.0
    val_max  = +3.0
    adj_T_C2 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_T_C2.connect( "value_changed", self.adjust_T_C2)
    adj_T_C2.emit( "value_changed")
    self.h_scale_T_C2 = gtk.HScale(adjustment=adj_T_C2)
    self.h_scale_T_C2.set_digits(2)
    self.scale_T_C2.add(self.h_scale_T_C2)
    self.Te_VB2.add(self.scale_T_C2)
    
    # --- Box for the adjustment bars
    self.scale_T_C3 = gtk.HBox()
    self.scale_T_C3.set_property("height-request", scale_height)
    self.scale_T_C3.add(gtk.Label("T_coef(3) (slope^3)"))
    val_init = DATA.Te_coef[4]
    val_min  = -3.0
    val_max  = +3.0
    adj_T_C3 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_T_C3.connect( "value_changed", self.adjust_T_C3)
    adj_T_C3.emit( "value_changed")
    self.h_scale_T_C3 = gtk.HScale(adjustment=adj_T_C3)
    self.h_scale_T_C3.set_digits(2)
    self.scale_T_C3.add(self.h_scale_T_C3)
    self.Te_VB2.add(self.scale_T_C3)
    
    # --- Box for the adjustment bars
    self.scale_T_C4 = gtk.HBox()
    self.scale_T_C4.set_property("height-request", scale_height)
    self.scale_T_C4.add(gtk.Label("T_coef(4) (ped-width)"))
    val_init = DATA.Te_coef[5]
    val_min  = 0.001
    val_max  = 0.3
    adj_T_C4 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_T_C4.connect( "value_changed", self.adjust_T_C4)
    adj_T_C4.emit( "value_changed")
    self.h_scale_T_C4 = gtk.HScale(adjustment=adj_T_C4)
    self.h_scale_T_C4.set_digits(3)
    self.scale_T_C4.add(self.h_scale_T_C4)
    self.Te_VB2.add(self.scale_T_C4)
    
    # --- Box for the adjustment bars
    self.scale_T_C5 = gtk.HBox()
    self.scale_T_C5.set_property("height-request", scale_height)
    self.scale_T_C5.add(gtk.Label("T_coef(5) (ped-pos)"))
    val_init = DATA.Te_coef[6]
    val_min  = 0.6
    val_max  = 1.5
    adj_T_C5 = gtk.Adjustment(value=val_init, lower=val_min, upper=val_max, step_incr=0.001, page_incr=0.001, page_size=0.001) 
    adj_T_C5.connect( "value_changed", self.adjust_T_C5)
    adj_T_C5.emit( "value_changed")
    self.h_scale_T_C5 = gtk.HScale(adjustment=adj_T_C5)
    self.h_scale_T_C5.set_digits(2)
    self.scale_T_C5.add(self.h_scale_T_C5)
    self.Te_VB2.add(self.scale_T_C5)
    
    self.Te_window.show_all()
    
  # --- Adjust fit with scales
  def adjust_T_0 (self, adj): DATA.Te_coef[0] = adj.value / (2.0*DATA.ne_coef[0]*mu_0*eV2Joules); self.update_Te_plot(self)
  def adjust_T_1 (self, adj): DATA.Te_coef[1] = adj.value / (2.0*DATA.ne_coef[0]*mu_0*eV2Joules); self.update_Te_plot(self)
  def adjust_T_C1(self, adj): DATA.Te_coef[2] = adj.value                                       ; self.update_Te_plot(self)
  def adjust_T_C2(self, adj): DATA.Te_coef[3] = adj.value                                       ; self.update_Te_plot(self)
  def adjust_T_C3(self, adj): DATA.Te_coef[4] = adj.value                                       ; self.update_Te_plot(self)
  def adjust_T_C4(self, adj): DATA.Te_coef[5] = adj.value                                       ; self.update_Te_plot(self)
  def adjust_T_C5(self, adj): DATA.Te_coef[6] = adj.value                                       ; self.update_Te_plot(self)
       
  # --- Update ne plot
  def update_Te_plot(self, data_thread):
    psi_fit, Te_fit  = get_fitted_curve(DATA.Te_coef,min(DATA.psi),max(DATA.psi))
    self.line_T.pop(0).remove()
    self.line_T = self.Te_ax.plot(psi_fit,Te_fit,'r',lw=2)
    self.Te_ax.set_title('Elec. temp. for JET pulse #' + str(DATA.pulse) + '\nfrom ' + str(DATA.t_min) + 's to ' + str(DATA.t_max) + 's')
    self.Te_ax.set_xlabel('psi [norm]')
    self.Te_ax.set_ylabel('Te [eV]')
    self.plots_update_Te()
    self.Te_window.show_all()
    self.update_coefs_print(self)
    
  # --- Need resize to update gtk???
  def plots_update_Te(self):
    gtk_class.window_partition = -gtk_class.window_partition
    xx, yy = self.Te_B1.size_request()
    yy = yy + gtk_class.window_partition
    self.Te_B1.set_size_request(xx,yy)
    xx, yy = self.Te_B2.size_request()
    yy = yy - gtk_class.window_partition
    self.Te_B2.set_size_request(xx,yy)

  # --- Close ne-plot and save ne-coefs
  def close_Te_window(self,widget):
    self.Te_window.destroy()
  
  
  
  
    
  # ----------------------------------------
  # --- Update printed coefs when printed
  def update_coefs_print(self,widget):
    # --- The coefs
    self.VB3.remove(self.rho_coefs_print)
    self.rho_coefs =                  " ! --- rho_coefs:\n"
    self.rho_coefs = self.rho_coefs + " central_density = "+str(DATA.ne_coef[0]*1.e-20)+"\n"
    self.rho_coefs = self.rho_coefs + " rho_0           = "+str(1.0)+"\n"
    self.rho_coefs = self.rho_coefs + " rho_1           = "+str(DATA.ne_coef[1]/DATA.ne_coef[0])+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(1)     = "+str(DATA.ne_coef[2])+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(2)     = "+str(DATA.ne_coef[3])+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(3)     = "+str(DATA.ne_coef[4])+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(4)     = "+str(DATA.ne_coef[5])+"\n"
    self.rho_coefs = self.rho_coefs + " rho_coef(5)     = "+str(DATA.ne_coef[6])+"\n"
    self.rho_coefs_print = gtk.Label(self.rho_coefs)
    self.rho_coefs_print.set_selectable(True)
    self.rho_coefs_print.set_justify(gtk.JUSTIFY_LEFT)
    self.VB3.add(self.rho_coefs_print)
    self.VB3.remove(self.T_coefs_print)
    self.T_coefs =                " ! --- T_coefs:\n"
    self.T_coefs = self.T_coefs + " ! --- central Te temperature: " + str(int(DATA.Te_coef[0])) + "eV\n"
    self.T_coefs = self.T_coefs + " T_0           = "+str(DATA.Te_coef[0]*2.0*DATA.ne_coef[0]*mu_0*eV2Joules)+"\n"
    self.T_coefs = self.T_coefs + " T_1           = "+str(DATA.Te_coef[1]*2.0*DATA.ne_coef[0]*mu_0*eV2Joules)+"\n"
    self.T_coefs = self.T_coefs + " T_coef(1)     = "+str(DATA.Te_coef[2])+"\n"
    self.T_coefs = self.T_coefs + " T_coef(2)     = "+str(DATA.Te_coef[3])+"\n"
    self.T_coefs = self.T_coefs + " T_coef(3)     = "+str(DATA.Te_coef[4])+"\n"
    self.T_coefs = self.T_coefs + " T_coef(4)     = "+str(DATA.Te_coef[5])+"\n"
    self.T_coefs = self.T_coefs + " T_coef(5)     = "+str(DATA.Te_coef[6])+"\n"
    self.T_coefs_print = gtk.Label(self.T_coefs)
    self.T_coefs_print.set_selectable(True)
    self.T_coefs_print.set_justify(gtk.JUSTIFY_LEFT)
    self.VB3.add(self.T_coefs_print)
    self.VB3.set_property("width-request", 10)
    self.window.show_all()







#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#-------------------------------------- MAIN SCRIPT EXECUTION -----------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------



# --- Main function to run the whole thing...
class Main():
  def __init__(self):
    data_thread = DATA()
    gtk_thread = gtk_class(data_thread)
    gtk.main()
      
Main()
