#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class HandyCell:
    '''An intracellular model without L-type calcium channel, Modified from Handy 2017'''
    def __init__(self, T = 20, dt = 0.001):

        # Initial Values
        self.ct0 = 35
        self.c0 = 0.05
        self.hh0 = None
        self.ip0 = 0.01

        # General Parameters
        self.gamma = 5.4054 
        self.delta = 0.2 

        # IP3R
        self.v_ip3r = 0.222
        self.v_in = 0.5
        self.d_1 = 0.13
        self.d_2 = 1.049
        self.d_3 = 0.9434
        self.d_5 = 0.08234
        self.a_2 = 0.04

        # SERCA
        self.v_serca = 0.9
        self.k_serca = 0.1

        # PMCA
        self.v_pmca = 10
        self.k_pmca = 2.5

        # IP3
        self.ip_decay = 0.04

        # Time constants
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    def i_ip3r(self, c, c_t, hh, ip):
        # IP3R [uM/s] (Li & Rinzel, 1996)
        mm_inf = ip / (ip + self.d_1)
        nn_inf = c / (c + self.d_5)
        return self.v_ip3r * mm_inf**3 * nn_inf**3 * hh**3 * ((c_t-c)*self.gamma - c)

    def i_serca(self, c):
        # SERCA [uM/s]
        return self.v_serca * c**1.75 / (c**1.75 + self.k_serca**1.75)

    def i_leak(self, c, c_t):
        # Leak from ER to cytosol [uM/s]
        v_leak = (- self.i_ip3r(self.c0, self.ct0, self.hh0, self.ip0) 
        + self.i_serca(self.c0)) / ((self.ct0-self.c0)*self.gamma - self.c0)
        return v_leak * ((c_t-c)*self.gamma - c)

    def i_pmca(self, c):
        # PMCA [uM/s]
        return self.v_pmca * c**2 / (c**2 + self.k_pmca**2)

    def i_add(self, c, c_t):
        # Additional fluxes from the extracellular space [uM/s]
        k_out = (self.v_in - self.i_pmca(self.c0)) / self.c0
        return self.v_in - k_out * c

    def hh_inf(self, c, ip):
        q_2 = self.d_2 * (ip + self.d_1)/(ip + self.d_3)
        return q_2 / (q_2 + c)

    def tau_hh(self, c, ip):
        q_2 = self.d_2 * (ip + self.d_1)/(ip + self.d_3)
        return 1 / (self.a_2 * (q_2 + c))

    def stim(self, t):
        # Stimulation
        if 20 <= t < 30:
            return 0.04
        else:
            return self.ip_decay * self.ip0

    def rhs(self, y, t):
        # Right-hand side formulation
        c, c_t, hh, ip = y

        dcdt = (self.i_ip3r(c, c_t, hh, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                + self.i_add(c, c_t)) * self.delta

        dctdt = (- self.i_pmca(c) + self.i_add(c, c_t)) * self.delta
        dhhdt = (self.hh_inf(c, ip) - hh) / self.tau_hh(c, ip)
        dipdt = self.stim(t) - self.ip_decay * ip

        return [dcdt, dctdt, dhhdt, dipdt]

    def step(self):
        # Time stepping
        self.hh0 = self.hh_inf(self.c0, self.ip0)
        
        y0 = [self.c0, self.ct0, self.hh0, self.ip0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=1000, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = HandyCell(T=100)
    sol = model.step()
    c = sol[:,0]
    c_t = sol[:,1]
    hh = sol[:,2]
    ip = sol[:,3]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot((c_t - c) * model.gamma, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(hh, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.show()

    # Plot the currents
    plt.figure()
    model.plot(model.i_ip3r(c, c_t, hh, ip), color='b')
    model.plot(model.i_serca(c), color = 'r')
    model.plot(model.i_pmca(c), color = 'g')
    model.plot(model.i_leak(c, c_t), color = 'y')
    model.plot(model.i_add(c, c_t), color = 'k')
    plt.legend(['i_ip3r', 'i_serca', 'i_pmca', 'i_leak', 'i_add'])
    plt.show()
