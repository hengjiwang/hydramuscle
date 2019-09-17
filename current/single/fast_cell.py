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
        self.d = 10e-4 # 10e-4 # [cm]
        self.F = 96485332.9 # [mA*s/mol]
        self.R = 8314000 # [uJ/K/mol]
        self.Temp = 308 # [K]


        self.c0 = 0.05
        self.v0 = -50 # (-40 to -60)
        self.n0 = 0.005911068856243796
        self.hv0 = 0.82324
        self.hc0 = 0.95238
        self.x0 = 0.05416670048123607
        self.z0 = 0.65052
        self.xa0 = None 
        self.xab10 = None 
        self.bx0 = None # 0.06951244510501192
        self.cx0 = None # 0.06889595335007676

        # Calcium leak
        self.tau_ex = 0.1 # [s]
        
        # CaL parameters
        self.g_cal = 0.0004 # [S/cm^2] 
        self.e_cal = 51
        self.k_cal = 1 # [uM]

        # CaT parameters
        self.g_cat = 0.0015
        self.e_cat = 51

        # KCNQ parameters
        self.g_kcnq = 0.0001 # [S/cm^2]
        self.e_k = -75 

        # KCa parameters
        self.g_kca = 0.00015
        self.e_k = -83.6
        self.pa = 0.2
        self.pb = 0.1

        # Background parameters
        self.g_bk = None
        self.e_bk = -91

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''CaL channel terms (Mahapatra 2018)'''
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
        return self.k_cal / (self.k_cal + c)

    def tau_n(self, v):
        # [s]
        return 0.000001 / (1 + np.exp(-(v+22)/308))

    def tau_hv(self, v):
        # [s]
        return 0.09 * (1 - 1 / ((1 + np.exp((v+14)/45)) * (1 + np.exp(-(v+9.8)/3.39))))

    def tau_hc(self):
        # [s]
        return 0.02

    '''T-type calcium channel terms (Mahapatra 2018)'''
    def i_cat(self, v, bx, cx):
        return self.g_cat * bx**2 * cx * (v - self.e_cat)

    def bx_inf(self, v):
        return 1 / (1 + np.exp(-(v+36.9)/6.6))

    def cx_inf(self, v):
        return 1 / (1 + np.exp((v+63.8)/5.3))

    def tau_bx(self, v):
        return (0.00045 + 0.0039 / (1 + ((v+66)/26)**2))

    def tau_cx(self, v):
        return (0.15 - (0.15/((1+ np.exp((v-417.43)/203.18))*(1+np.exp(-(v+61.11)/8.07)))))
    
    '''KCNQ1 channel terms (Mahapatra 2018)'''
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

    '''BK channels (Tong 2011)'''
    def i_kca(self, v, c, xa, xab1):

        def ia(xa, v):
            return xa * (v - self.e_k)
        
        def iab1(xab1, v):
            return xab1 * (v - self.e_k)

        return self.g_kca * (self.pa * ia(xa, v) + self.pb * iab1(xab1, v))
        
    def ssa(self, v, c):
        
        def za(c):
            return 8.38 / (1 + ((1000*c + 1538.29)/739.06)**2) - 0.749 / (1 + ((1000*c-0.063)/0.162)**2)

        def v05a(c):
            return 5011.47 / (1 + ((1000*c+0.238)/0.000239)**0.423) - 37.51

        return 1 / (1 + np.exp(-za(c) * self.F * (v - v05a(c)) / self.R / self.Temp))

    def tau_a(self, v):
        return 0.001 * 2.41 / (1 + ((v-158.78)/-52.15)**2)

    def ssab1(self, v, c):

        def zab1(c):
            return 1.4 / (1 + ((1000*c + 228.71)/684.95)**2) - 0.681 / (1 + ((1000*c - 0.219)/0.428)**2)

        def v05ab1(c):
            return 8540.23 / (1 + ((1000*c + 0.401)/0.00399)**0.668) - 109.28

        return 1 / (1 + np.exp(- zab1(c) * self.F * (v - v05ab1(c)) / self.R / self.Temp))

    def tau_ab1(self, v):
        return 0.001 * 13.8 / (1 + ((v-153.02)/66.5)**2)

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        g_bk = - (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) \
        + self.i_cat(self.v0, self.bx0, self.cx0) \
        + self.i_kcnq(self.v0, self.x0, self.z0) \
        + self.i_kca(self.v0, self.c0, self.xa0, self.xab10))/(self.v0 - self.e_bk)

        return  g_bk * (v - self.e_bk)

    '''Calcium terms'''
    def r_ex(self, c):
        # [uM/s]
        return (c-self.c0)/self.tau_ex

    '''Stimulation'''
    def stim(self, t):
        if 1 <= t < 1.01:
            return 1
        else:
            return 0

    '''Numerical terms'''
    def rhs(self, y, t):
        # Right-hand side function
        c, v, n, hv, hc, x, z, xa, xab1, bx, cx = y
        dcdt = - self.r_ex(c) \
            - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d) \
            + 1e9 * self.i_cal(self.v0, self.n0, self.hv0, self.hc0) / (2 * self.F * self.d) \
            - 1e9 * self.i_cat(v, bx, cx) / (2 * self.F * self.d) \
            + 1e9 * self.i_cat(self.v0, self.bx0, self.cx0) / (2 * self.F * self.d)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kcnq(v, x, z) + self.i_kca(v, c, xa, xab1) + self.i_bk(v) - 0.001 * self.stim(t))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)
        dxadt = (self.ssa(v, c) - xa)/self.tau_a(v)
        dxab1dt = (self.ssab1(v, c) - xab1)/self.tau_ab1(v)
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)

        return [dcdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt, dxadt, dxab1dt, dbxdt, dcxdt]

    def step(self):
        # Time stepping

        self.n0 = self.n_inf(self.v0)
        self.xa0 = self.ssa(self.v0, self.c0)
        self.xab10 = self.ssab1(self.v0, self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        print(self.bx0, self.cx0)


        y0 = [self.c0, self.v0, self.n0, self.hv0, self.hc0, self.x0, self.z0, self.xa0, self.xab10, self.bx0, self.cx0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Visualize results'''
    def plot(self, a, tmin=0, tmax=20, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
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
    xa = sol[:, 7]
    xab1 = sol[:, 8]
    bx = sol[:, 9]
    cx = sol[:, 10]

    # Plot the results
    plt.figure()
    plt.subplot(421)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(422)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(423)
    model.plot(model.i_cal(v, n , hv, hc), ylabel='i_cal[mA/cm^2]')
    plt.subplot(424)
    model.plot(model.i_cat(v, bx, cx), ylabel='i_cat[mA/cm^2]')
    plt.subplot(425)
    model.plot(model.i_kcnq(v, x, z), ylabel='i_kncq[mA/cm^2]')
    plt.subplot(426)
    model.plot(model.i_kca(v, c, xa, xab1), ylabel='i_kca[mA/cm^2]')
    plt.subplot(427)
    model.plot([0.004 * model.stim(t) for t in model.time], ylabel='i_stim[mA/cm^2]', color = 'r')
    plt.show()
    
    

    
