# -------------------------------------------------
# Import all modules
# -------------------------------------------------
try:
    import sys
    import sys, traceback
    import numpy
    import pylab
    import pyuda
    import py_flush as Flush
    from scipy import optimize

    # from math import pi, sin, cos, sqrt, exp, atan2, tanh, cosh
    import matplotlib.pyplot as plt
    import matplotlib
    import os
except:
    print("-------failed to import module!---------")
    traceback.print_exc(file=sys.stdout)
    sys.exit(127)


def residual(a, y, x):
    err = y - curve(x, a)
    return err


def curve(x, a):
    poly = 1.0 + a[2] * x + a[3] * x * x + a[4] * x * x * x
    atanh = 0.5 - 0.5 * numpy.tanh((x - a[6]) / a[5])
    return (a[0] - a[1]) * poly * atanh + a[1]


# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():

    pulse = 30356
    t_min = 0.210
    t_max = 0.280
    t_mid = (t_min + t_max) / 2.0

    # --- Prepare plots
    matplotlib.rcParams.update({"font.size": 24})
    matplotlib.rcParams.update({"font.weight": "bold"})
    matplotlib.rcParams.update({"legend.fontsize": 18})

    # Create a client instance
    client = pyuda.Client()

    # Retrieve data
    RR = client.get("ayc_R", pulse)
    ne = client.get("ayc_ne", pulse)
    Te = client.get("ayc_te", pulse)
    Ip = client.get("ip", pulse)
    Bt = client.get("bt", pulse)
    R0 = client.get("Rgeo", pulse)
    qax = client.get("efm_q_axis", pulse)
    q95 = client.get("efm_q_95", pulse)
    P_nbi = client.get("anb_tot_sum_power", pulse)

    # --- Get list of eqdsk files
    eqdsks = os.listdir("./eqdsks/")
    n_eqdsks = numpy.size(eqdsks)
    eqdsk_times = numpy.zeros(n_eqdsks)
    for i in range(n_eqdsks):
        eqdsk = eqdsks[i]
        eqdsk_times[i] = float(eqdsk[10:18])

    # --- Join all profiles into one dataset for fitting
    ne_all = numpy.array([])
    Te_all = numpy.array([])
    R_all = numpy.array([])
    psi_all = numpy.array([])
    for i_time in range(numpy.size(ne.time.data)):
        R_tmp = RR.data[i_time]
        ne_tmp = ne.data[i_time]
        Te_tmp = Te.data[i_time]
        time = ne.time.data[i_time]
        if (time < t_min) or (time > t_max):
            continue
        # --- Get psi from Flush
        int_time = int(time * 1.0e3)
        i_min = 0
        diff_min = 1.0e10
        for i in range(n_eqdsks):
            if abs(eqdsk_times[i] - int_time) < diff_min:
                diff_min = abs(eqdsk_times[i] - int_time)
                i_min = i
        str_time = str(int(eqdsk_times[i_min]))
        gfile = "eqdsks/g0" + str(pulse) + ".00" + str_time
        time_flush, ier = Flush.flushinit(62, 0, 0.0, 0, 0, "", gfile, 0)
        if ier != 0:
            print("Flushinit failed : ", ier)
        n_flush = numpy.size(R_tmp)
        R_flush = numpy.zeros(n_flush)
        for i in range(n_flush):
            R_flush[i] = R_tmp[i] * 1.0e2
        Z_flush = numpy.linspace(0.0, 0.0, n_flush)
        f_flush, ier = Flush.Flush_getFlux(n_flush, R_flush, Z_flush)

        for i in range(numpy.size(R_tmp)):
            R_tmp2 = R_tmp[i]
            ne_tmp2 = ne_tmp[i]
            Te_tmp2 = Te_tmp[i]
            psi_tmp2 = f_flush[i]
            if not ((ne_tmp2 <= 0.0) or (ne_tmp2 >= 0.0)):
                ne_tmp2 = 0.0
            if not ((Te_tmp2 <= 0.0) or (Te_tmp2 >= 0.0)):
                Te_tmp2 = 0.0
            ne_all = numpy.append(ne_all, ne_tmp2)
            Te_all = numpy.append(Te_all, Te_tmp2)
            R_all = numpy.append(R_all, R_tmp2)
            psi_all = numpy.append(psi_all, psi_tmp2)

    # --- Clean temperature data
    for i in range(numpy.size(Te_all)):
        if Te_all[i] > 4000:
            Te_all[i] = 0.0
        if (psi_all[i] > 1.0) and (Te_all[i] > 500):
            Te_all[i] = 0.0

    # --- Violent clean!
    if 1 == 1:
        for i in range(numpy.size(Te_all)):
            if psi_all[i] > 1.1:
                Te_all[i] = 0.0
            if psi_all[i] > 1.1:
                ne_all[i] = 0.0

    # --- Fit density data
    a0 = [max(ne_all), 0.0, 0.0, 0.0, 0.0, 0.01, 0.9]
    a1 = optimize.leastsq(residual, a0, args=(ne_all, psi_all), maxfev=2000)
    a1 = a1[0]
    a1[1] = a1[0] / 100.0
    n_0 = a1[0]
    # --- Fit temperature data
    b0 = [max(Te_all), 10.0, -1.0, 0.0, 0.0, 0.01, 0.98]
    b1 = optimize.leastsq(residual, b0, args=(Te_all, psi_all), maxfev=2000)
    b1 = b1[0]
    # --- Plot fitted data
    n_fit = 1000
    ne_fit = numpy.zeros(n_fit)
    Te_fit = numpy.zeros(n_fit)
    psi_fit = numpy.linspace(0.0, 1.2, n_fit)
    for i in range(n_fit):
        ne_fit[i] = curve(psi_fit[i], a1)
        Te_fit[i] = curve(psi_fit[i], b1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(psi_all, ne_all, "b+", linewidth=2, markersize=2)
    ax1.plot(psi_fit, ne_fit, "r", linewidth=2, markersize=2)
    ax1.set_xlabel("psi [norm]")
    ax1.set_ylabel("ne")
    ax1.set_title("ne data")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(psi_all, Te_all, "b+", linewidth=2, markersize=2)
    ax1.plot(psi_fit, Te_fit, "r", linewidth=2, markersize=2)
    ax1.set_xlabel("psi [norm]")
    ax1.set_ylabel("Te")
    ax1.set_title("Te data")
    pylab.show()

    print("central_density = " % (n_0 / 1.0e20))
    print("rho_0       = %f" % (a1[0] / n_0))
    print("rho_1       = %f" % (a1[1] / n_0))
    print("rho_coef(1) = %f" % (a1[2]))
    print("rho_coef(2) = %f" % (a1[3]))
    print("rho_coef(3) = %f" % (a1[4]))
    print("rho_coef(4) = %f" % (a1[5]))
    print("rho_coef(5) = %f" % (a1[6]))

    return

    # Examine returned data object
    print(Ip.label)
    print(Ip.units)
    print(Ip.data)
    print(Ip.time.label)
    print(Ip.time.units)
    print(Ip.time.data)

    central_density = 0.46e20
    rho_norm = central_density * 3.32e-27
    mu_0 = 1.2566370614e-6
    t_norm = sqrt(mu_0 * rho_norm)
    TT_norm = 1.0 / (mu_0 * central_density)
    eV2Joules = 1.6e-19
    mu0 = 4.0 * pi * 1.0e-7

    # prof 1
    d1 = numpy.zeros(10)
    t1 = numpy.zeros(10)
    d1[0] = 1.000000
    t1[0] = 0.122482
    d1[1] = 0.100000
    t1[1] = 0.001225
    d1[2] = -0.577453
    t1[2] = -1.100059
    d1[3] = -0.640000
    t1[3] = -0.066370
    d1[4] = 0.650000
    t1[4] = 0.475000
    d1[5] = 0.020000
    t1[5] = 0.038000
    d1[6] = 0.95
    t1[6] = 0.96129

    # prof 2
    d2 = numpy.zeros(10)
    t2 = numpy.zeros(10)
    d2[0] = 0.7
    t2[0] = 0.122482
    d2[1] = 0.07
    t2[1] = 0.001225
    d2[2] = -0.15
    t2[2] = -1.100059
    d2[3] = -0.5
    t2[3] = -0.066370
    d2[4] = 0.3
    t2[4] = 0.475000
    d2[5] = 0.020000
    t2[5] = 0.038000
    d2[6] = 0.95
    t2[6] = 0.96129

    # --- Plot fitted profiles
    npsi = 300
    psi_large = numpy.linspace(0.0, 1.1, npsi)
    D1 = numpy.array([])
    T1 = numpy.array([])
    P1 = numpy.array([])
    D2 = numpy.array([])
    T2 = numpy.array([])
    P2 = numpy.array([])
    for i in range(0, npsi):
        D1 = numpy.append(D1, curve(psi_large[i], d1))
        T1 = numpy.append(T1, curve(psi_large[i], t1))
        D2 = numpy.append(D2, curve(psi_large[i], d2))
        T2 = numpy.append(T2, curve(psi_large[i], t2))
        P1 = numpy.append(P1, curve(psi_large[i], d1) * curve(psi_large[i], t1))
        P2 = numpy.append(P2, curve(psi_large[i], d2) * curve(psi_large[i], t2))

    dP1 = numpy.zeros(npsi)
    dP2 = numpy.zeros(npsi)
    for i in range(1, npsi):
        dP1[i] = -(P1[i] - P1[i - 1]) / (psi_large[i] - psi_large[i - 1])
        dP2[i] = -(P2[i] - P2[i - 1]) / (psi_large[i] - psi_large[i - 1])

    if 1:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(psi_large, D1 * central_density, "b", linewidth=2)
        ax1.plot(psi_large, D2 * central_density, "r", linewidth=2)
        pylab.xlabel("psi (normalised)")
        pylab.ylabel("ne [au]")
        pylab.title("Density")
        pylab.xlim([0.0, 1.1])
        pylab.ylim([0.0, central_density])
        ax1.legend(["old", "new"])

    if 0:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(psi_large, T1 / T1[0], "b", linewidth=2)
        ax1.plot(psi_large, T2 / T2[0], "r", linewidth=2)
        pylab.xlabel("psi (normalised)")
        pylab.ylabel("Te [au]")
        pylab.title("Temperature")
        pylab.xlim([0.0, 1.1])
        pylab.ylim([0.0, 1.1])
        ax1.legend(["old", "new"])

    if 1:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(
            psi_large, T1 / (central_density * mu0 * eV2Joules) / 2.0, "b", linewidth=2
        )
        ax1.plot(
            psi_large, T2 / (central_density * mu0 * eV2Joules) / 2.0, "r", linewidth=2
        )
        pylab.xlabel("psi (normalised)")
        pylab.ylabel("Te [eV]")
        pylab.title("Temperature")
        pylab.xlim([0.0, 1.1])
        ax1.legend(["old", "new"])
        for i in range(numpy.size(psi_large)):
            print(
                "%f  %e  %f"
                % (
                    psi_large[i],
                    D1[i] * central_density,
                    T1[i] / (central_density * mu0 * eV2Joules) / 2.0,
                )
            )

    if 0:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(psi_large, dP1 / max(max(dP1), max(dP2)), "b", linewidth=2)
        ax1.plot(psi_large, dP2 / max(max(dP1), max(dP2)), "r", linewidth=2)
        pylab.xlabel("psi (normalised)")
        pylab.ylabel("dP [au]")
        pylab.title("Pressure Gradient")
        pylab.xlim([0.0, 1.1])
        pylab.ylim([0.0, 1.1])
        ax1.legend(["old", "new"])

    if 0:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(psi_large, D1 / D1[0], "k", linewidth=2)
        ax1.plot(psi_large, T1 / T1[0], "r", linewidth=2)
        ax1.plot(psi_large, D2 / D2[0], "k--", linewidth=2)
        ax1.plot(psi_large, T2 / T2[0], "r--", linewidth=2)
        pylab.xlabel("psi (normalised)")
        pylab.ylabel("Te [au]")
        pylab.title("Both")
        pylab.xlim([0.0, 1.1])
        pylab.ylim([0.0, 1.1])
        ax1.legend(["old ne", "old Te", "new ne", "new Te"])

    pylab.show()
    return


def plot_flush(gfile):

    # --- Initialise Flush
    time, ier = Flush.flushinit(62, 0, 0.0, 0, 0, "", gfile, 0)
    if ier != 0:
        print("Flushinit failed : ", ier)

    # ----Plot the first wall----
    nlim, rlim, zlim, ier = Flush.Flush_readFirstWall()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(rlim[0:nlim], zlim[0:nlim], c="g", linewidth=3)

    # ----Get Axis and Xpoints to get normalised value of second Xpoint----
    nXpoint, rXpoint, zXpoint, fXpoint, ier = Flush.Flush_getAllXpoints()
    rAxis, zAxis, fAxis, ier = Flush.Flush_getMagAxis()
    psi_bnd2 = 1000.0
    psi_bnd3 = 1000.0

    # ----Take a range of flux values----
    npsi = 60
    accuracy = 5.0

    # ----Call routine----
    psisurs = numpy.linspace(0.0, 2.5, npsi)
    col = "b"
    print("Calling :  Flush_getFluxSurfaces...")
    npieces, npoints, rsurf, zsurf, ier = Flush.Flush_getFluxSurfaces(
        npsi, psisurs, accuracy
    )
    print("Finished:   ", ier)
    # ----Plot surfaces----
    if ier == 0:
        for i in range(0, npsi):
            for j in range(0, npieces[i]):
                if npoints[i][j] > 0:
                    ax1.plot(
                        rsurf[i][j][0 : npoints[i][j]],
                        zsurf[i][j][0 : npoints[i][j]],
                        c=col,
                    )
    col = "r"
    print("Calling :  Flush_getMainXpointSurface...")
    npoints, rsurf, zsurf, ier = Flush.Flush_getMainXpointSurface(accuracy)
    print("Finished:   ", ier)
    # ----Plot surfaces----
    if ier == 0:
        if npoints > 0:
            ax1.plot(rsurf[0:npoints], zsurf[0:npoints], c=col)
    print("Calling :  Flush_getSecondXpointSurface...")
    npoints, rsurf, zsurf, ier = Flush.Flush_getSecondXpointSurface(accuracy)
    print("Finished:   ", ier)
    # ----Plot surfaces----
    if ier == 0:
        if npoints > 0:
            ax1.plot(rsurf[0:npoints], zsurf[0:npoints], c=col)

    pylab.axis("equal")
    pylab.show()

    return


##################################################################
################### Execution Routine ############################
##################################################################
if __name__ == "__main__":
    """Main program"""
    try:
        main()
    except:
        print("-------unhandled exception!---------")
        traceback.print_exc(file=sys.stdout)
        sys.exit(127)

