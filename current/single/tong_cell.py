#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class TongCell:
    '''Following Tong et al. 2011'''
    def __init__(self, T = 20, dt = 0.001):
        # General parameters
        self.F = 96485 # [C/mol]
        self.R = 8314 # [mJ/K/mol]
        self.Temp = 308 # [K]

        # Calcium leak
        self.tau_ex = 0.1
        
        # CaL parameters
        self.g_cal = 0.6 # [nS/pF]
        self.e_cal = 45
        self.k_dcal = 1.0

        # CaT parameters
        self.g_cat = 0.058 # [nS/pF]
        self.e_cat = 42

        # KCa parameters
        self.g_kca = 0.8 # [nS/pF]
        self.e_k = -83.6
        self.pa = 0.2
        self.pb = 0.1

        # Background parameters
        self.e_bk = -83.6

        # Initial values
        self.v0 = -50
        self.c0 = 0.05
        self.d0 = 0.01798620996209156
        self.f10 = 0.8473913351573689
        self.f20 = 0.8473913351573689
        self.b0 = 0.6054288693090248
        self.g0 = 0.026875236839296875
        self.xa0 = 0.0935111953309457
        self.xab10 =  0.8787194884600014

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''L-type calcium current terms'''
    def i_cal(self, v, c, d, f1, f2):

        def f_ca(c):
            return 1 / (1 + (c / self.k_dcal)**4)

        return self.g_cal * d**2 * f_ca(c) * (0.8 * f1 + 0.2 * f2) * (v - self.e_cal)

    def d_inf(self, v):
        return 1 / (1 + np.exp(-(v+22)/7))

    def f_inf(self, v):
        return 1 / (1 + np.exp((v+38)/7))

    def tau_d(self, v):
        return 0.001 * (2.29 + 5.7/(1 + ((v+29.97)/9)**2))

    def tau_f1(self, v):
        return 0.001 * 12

    def tau_f2(self, v):
        return 0.001 * 90.97 * (1 - 1 / ((1 + np.exp((v+13.96)/45.38))*(1 + np.exp(-(v+9.5)/3.39))))

    '''T-type calcium current terms'''
    def i_cat(self, v, b, g):
        return self.g_cat * b**2 * g * (v - self.e_cat)

    def b_inf(self, v):
        return 1 / (1 + np.exp(-(v+54.23)/9.88))

    def g_inf(self, v):
        return 0.02 + 0.98 / (1 + np.exp((v+72.98)/4.64))

    def tau_b(self, v):
        return 0.001 * (0.45 + 3.9 / (1 + ((v+66)/26)**2))

    def tau_g(self, v):
        return 0.001 * (150 - 150/((1+ np.exp((v-417.43)/203.18))*(1+np.exp(-(v+61.11)/8.07))))

    '''Calcium-activation potassium current terms'''
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
        g_bk = - (self.i_cal(self.v0, self.c0, self.d0, self.f10, self.f20) \
        + self.i_cat(self.v0, self.b0, self.g0) \
        + self.i_kca(self.v0, self.c0, self.xa0, self.xab10)) / (self.v0 - self.e_bk)
        
        return  g_bk * (v - self.e_bk)

    '''Stimulation'''
    def stim(self, t):
        if 1 <= t < 1.02:
            return 1
        else:
            return 0

    '''Calcium terms'''
    def r_ex(self, c):
        # [uM/s]
        return (c-self.c0)/self.tau_ex


    '''Numerical terms'''
    def rhs(self, y, t):
        # Right-hand side formulation
        c, v, d, f1, f2, b, g, xa, xab1 = y

        dcdt = - self.r_ex(c) + (- self.i_cal(v, c, d, f1, f2) \
            - self.i_cat(v, b, g) \
            + self.i_cal(self.v0, self.c0, self.d0, self.f10, self.f20) \
            + self.i_cat(self.v0, self.b0, self.g0)) * 3.1e-04
        
        dvdt = - 1000 * (self.i_cal(v, c, d, f1, f2) + self.i_cat(v, b, g) + self.i_kca(v, c, xa, xab1) + self.i_bk(v) - 1.5*self.stim(t))
        dddt = (self.d_inf(v) - d) / self.tau_d(v)
        df1dt = (self.f_inf(v) - f1) / self.tau_f1(v)
        df2dt = (self.f_inf(v) - f2) / self.tau_f2(v)
        dbdt = (self.b_inf(v) - b) / self.tau_b(v)
        dgdt = (self.g_inf(v) - g) / self.tau_g(v)
        dxadt = (self.ssa(v,c) - xa) / self.tau_a(v)
        dxab1dt = (self.ssab1(v,c) - xab1) / self.tau_ab1(v)

        return [dcdt, dvdt, dddt, df1dt, df2dt, dbdt, dgdt, dxadt, dxab1dt]

    def step(self):
        # Time stepping
        y0 = [self.c0, self.v0, self.d0, self.f10, self.f20, self.b0, self.g0, self.xa0, self.xab10]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=2, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = TongCell()
    sol = model.step()
    c = sol[:, 0]
    v = sol[:, 1]
    d = sol[:, 2]
    f1 = sol[:, 3]
    f2 = sol[:, 4]
    b = sol[:, 5]
    g = sol[:, 6]
    xa = sol[:, 7]
    xab1 = sol[:, 8]

    # Plot the results
    plt.figure()
    plt.subplot(421)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(422)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(423)
    model.plot(model.i_cal(v, c, d, f1, f2), ylabel='i_cal[pA/pF]')
    plt.subplot(424)
    model.plot(model.i_cat(v, b, g), ylabel='i_cat[pA/pF]')
    plt.subplot(425)
    model.plot(model.i_kca(v, c, xa, xab1), ylabel='i_kca[pA/pF]')
    plt.subplot(426)
    model.plot(model.i_bk(v), ylabel='i_bk[pA/pF]')
    plt.subplot(427)
    model.plot([model.stim(t) for t in model.time], ylabel='i_stim[pA/pF]', color = 'r')
    plt.show()

