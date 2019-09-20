#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class YochumCell:
    '''Following Yochum 2016'''
    def __init__(self, T = 20, dt = 0.001):
        # General parameters
        self.c_m = 1e-3 # [mF/cm^2]
        self.R = 8.314 # [J/K/mol]
        self.F = 96.487 # [kC/mol]
        self.Temp = 295 # [K]
 
        # Calcium current parameters
        self.g_ca = 0.02694061 # [mS/cm^2]
        self.v_ca = -20.07451779 # [mV]
        self.r_ca = 5.97139101 # [mV]
        self.j_back = 0.02397327 # [uA/cm^2]
        self.co = 3000 # [uM]

        # Potassium current parameters
        self.g_k = 0.064 # [mS/cm^2]
        self.e_k = -83 # [mV]
        self.g_kca = 0.08 # [mS/cm^2]
        self.kai = 10 # [uM]

        # Leak parameters
        self.g_l = 0.0055 # [mS/cm^2]
        self.e_l = -20 # [mV]

        # Calcium dynamics parameters
        self.k_ca = 10 # [s-1]
        self.f_c = 0.4
        self.alpha = 40 # [uM*cm^2/uC]

        # Initial values
        self.c0 = 0.1
        self.v0 = - 50
        self.n0 = 0.079257

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''Calcium terms'''
    def i_ca(self, v, c):
        # Calcium current (L-type, T-type and background)

        e_ca = self.R * self.Temp / (2 * self.F) * np.log(self.co / c)

        return self.g_ca * 1 / (1 + np.exp((self.v_ca - v)/self.r_ca)) * (v - e_ca) + self.j_back

    '''Voltage-gated Potassium terms'''
    def i_k(self, v, n):
        # Voltage-gated potassium current
        return self.g_k * n * (v - self.e_k)

    def n_inf(self, v):
        return 1 / (1 + np.exp((4.2-v)/21.1))

    def tau_n(self, v):
        return 0.02375 * np.exp(-v/72.15)

    '''CaK terms'''
    def i_kca(self, v, c):
        return self.g_kca * c**2 / (c**2 + self.kai**2) * (v - self.e_k)

    '''Leak channel'''
    def i_l(self, v):

        e_l = self.v0 - (- self.i_ca(self.v0, self.c0) \
             - self.i_k(self.v0, self.n0) - self.i_kca(self.v0, self.c0)) / self.g_l

        return self.g_l * (v - e_l)

    '''Stimulation'''
    def i_stim(self, t):
        if 1 <= t < 1.4:
            return 0 # 0.1175
        else:
            return 0

    '''Numerical terms'''
    def rhs(self, y, t):

        c, v, n = y

        dcdt = self.f_c * (-self.alpha*self.i_ca(v, c) - self.k_ca * c)
        dvdt = 1/self.c_m * (self.i_stim(t) - self.i_ca(v, c) - self.i_k(v, n) - self.i_kca(v, c) - self.i_l(v))
        dndt = (self.n_inf(v) - n) / self.tau_n(v)

        return [dcdt, dvdt, dndt]

    def step(self):
        
        self.n0 = self.n_inf(self.v0)
        self.k_ca = -self.alpha*self.i_ca(self.v0, self.c0) / self.c0

        y0 = [self.c0, self.v0, self.n0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Visualize results'''
    def plot(self, a, tmin=0, tmax=20, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = YochumCell(20)
    sol = model.step()
    c = sol[:, 0]
    v = sol[:, 1]
    n = sol[:, 2]

    plt.figure()
    plt.subplot(311)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(312)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(313)
    model.plot(n, ylabel='n')
    plt.show()

    