#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class FastCell:
    '''An intracellular model for calcium influx from extracellular space'''
    def __init__(self, T = 20, dt = 0.001):
        # General parameters
        self.c_m = 1e-6 # [F/cm^2]
        self.A_cyt = 4e-5 # [cm^2]
        self.V_cyt = 6e-9 # [cm^3]
        self.d = 0.5e-4 # [cm]
        self.F = 96485332.9 # [mA*s/mol]
        self.c0 = 0.05
        self.v0 = -50 # (-40 to -60)
        self.n0 = 0.00591
        self.hv0 = 0.82324
        self.hc0 = 0.95238
        self.x0 = 0.05417
        self.z0 = 0.65052

        # Calcium leak
        self.tau_ex = 0.1 # [s]
        
        # CaL parameters
        self.g_cal = 0.0006 # [S/cm^2] 
        self.e_cal = 51
        self.ki = 1 # [uM]

        # KCNQ parameters
        self.g_kcnq = 0.0001 # [S/cm^2]
        self.e_k = -75 

        # Background parameters
        self.g_bk = None
        self.e_bk = -91

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''CaL channel terms'''
    def i_cal(self, v, n, hv, hc):
        # L-type calcium channel [mA/cm^2]
        return self.g_cal * n**2 * hv * hc * (v - self.e_cal)

    def n_inf(self, v):
        # [-]
        return 1 / (1 + np.exp(-(v+9)/8))

    def hv_inf(self, v):
        # [-]
        return 1 / (1 + np.exp((v+30)/13))

    def hc_inf(self, c):
        # [-]
        return self.ki / (self.ki + c)

    def tau_n(self, v):
        # [s]
        return 0.000001 / (1 + np.exp(-(v+22)/308))

    def tau_hv(self, v):
        # [s]
        return 0.09 * (1 - 1 / ((1 + np.exp((v+14)/45)) * (1 + np.exp(-(v+9.8)/3.39))))

    def tau_hc(self):
        # [s]
        return 0.02
    
    '''KCNQ1 channel terms'''
    def i_kcnq(self, v, x, z):
        # KCNQ1 channel [mA/cm^2]
        return self.g_kcnq * x**2 * z * (v - self.e_k)

    def x_inf(self, v):
        # [-]
        return 1 / (1 + np.exp(-(v+7.1)/15))

    def z_inf(self, v):
        # [-]
        return 0.55 / (1 + np.exp((v+55)/9)) + 0.45

    def tau_x(self, v):
        # [s]
        return 0.001 / (1 + np.exp((v+15)/20))

    def tau_z(self, v):
        # [s]
        return 0.01 * (200 + 10 / (1 + 2 * ((v+56)/120)**5))

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        g_bk = - (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) \
        + self.i_kcnq(self.v0, self.x0, self.z0))/(self.v0 - self.e_bk)
        return  g_bk * (v - self.e_bk)

    '''Calcium terms'''
    def r_ex(self, c):
        # [uM/s]
        return (c-self.c0)/self.tau_ex

    '''Stimulation'''
    def stim(self, t):
        if 1 <= t < 1.1:
            return 1
        else:
            return 0

    '''Numerical terms'''
    def rhs(self, y, t):
        # Right-hand side function
        c, v, n, hv, hc, x, z = y
        dcdt = - self.r_ex(c) - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_kcnq(v, x, z) + self.i_bk(v) - 0.0005 * self.stim(t))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)

        return [dcdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt]

    def step(self):
        # Time stepping
        y0 = [self.c0, self.v0, self.n0, self.hv0, self.hc0, self.x0, self.z0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Visualize results'''
    def plot(self, a, tmin=0, tmax=2, xlabel = 'time[s]', ylabel = None):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = FastCell()
    sol = model.step()
    c = sol[:, 0]
    v = sol[:, 1]
    n = sol[:, 2]
    hv = sol[:, 3]
    hc = sol[:, 4]
    x = sol[:, 5]
    z = sol[:, 6]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(222)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(223)
    model.plot(model.i_cal(v, n , hv, hc), ylabel='i_cal[mA/cm^2]')
    plt.subplot(224)
    model.plot(model.i_kcnq(v, x, z), ylabel='i_kncq[mA/cm^2]')
    plt.show()
    
    

    
