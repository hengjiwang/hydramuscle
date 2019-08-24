#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HoferCell:
    '''An intracellular model following Hofer 2002'''

    def __init__(self, T = 60, dt = 0.001):
        self.k1 = 0.0004
        self.k2 = 0.08
        self.ka = 0.2
        self.kip = 0.3
        self.k3 = 0.5
        self.v40 = 0.025
        self.v41 = 0.2
        self.kr = 1
        self.k5 = 0.5
        self.k6 = 4
        self.ki = 0.2
        self.kg = 0.1 # unknown
        self.a0 = 1 # 1e-3 - 10
        self.v7 = 0.04 # 0 - 0.05
        self.v8 = 4e-4
        self.kca = 0.3
        self.k9 = 0.08
        self.beta = 20

        self.c0 = 0.05
        self.s0 = 100 # 60
        self.r0 = 0.94
        self.ip0 = 0.01

        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''Calcium terms'''
    def i_rel(self, c, s, ip, r):
        # Release from ER, including IP3R and leak term [uM/s]
        return (self.k2 * r * c**2 * ip**2 / (self.ka**2 + c**2) / (self.kip**2 + ip**2)) * (s - c)

    def i_serca(self, c):
        # SERCA [uM/s]
        # v_serca = 2
        # k_serca = 0.1
        # return v_serca * c / (c + k_serca)
        return self.k3 * c

    def i_leak(self, c, s):
        k1 = (self.i_serca(self.c0) - self.i_rel(self.c0, self.s0, self.ip0, self.r0)) / (self.s0 - self.c0)
        return k1 * (s - c)

    def i_in(self, c, ip):
        # Calcium entry rate [uM/s]
        return self.v40 + self.v41 * ip**2 / (self.kr**2 + ip**2)

    def i_pmca(self, c):
        # PMCA [uM/s]
        k_pmca = 2.5
        v_pmca = 4
        return 0 * v_pmca * c**2 / (c**2 + k_pmca**2)

    def i_out(self, c):
        # Additional eflux [uM/s]
        k5 = (self.i_in(self.c0, self.ip0) - self.i_pmca(self.c0)) / self.c0
        return k5 * c

    '''IP3R terms'''
    def v_r(self, c, r):
        # Rates of receptor inactivation and recovery [1/s]
        return self.k6 * (self.ki**2 / (self.ki**2 + c**2) - r)

    '''IP3 terms'''
    def i_plcb(self, v8):
        # Agonist-controlled PLC-beta activity [uM/s]
        return v8 * 1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0

    def i_plcd(self, c):
        # PLC-delta activity [uM/s]
        return 0 * self.v7 * c**2 / (self.kca**2 + c**2)

    def i_deg(self, ip):
        # IP3 degradion [uM/s]
        k9 = (self.i_plcb(self.v8) + self.i_plcd(self.c0)) / self.ip0
        return k9 * ip

    '''Stimulation'''
    def stim(self, t):
        # Stimulation
        if 10 <= t < 14:
            return 1
        else:
            return self.v8

    '''Numerical terms'''
    def rhs(self, y, t):
        # Right-hand side formulation
        c, s, r, ip = y

        dcdt = self.i_rel(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in(c, ip) - self.i_pmca(c) - self.i_out(c)
        dsdt = self.beta * (self.i_serca(c) - self.i_rel(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.stim(t)) + self.i_plcd(c) - self.i_deg(ip)

        return [dcdt, dsdt, drdt, dipdt]

    def step(self):
        # Time stepping    
        y0 = [self.c0, self.s0, self.r0, self.ip0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Plot'''
    def plot(self, a, tmin=0, tmax=100, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":
    model = HoferCell(100)
    sol = model.step()
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(r, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.show()

    # Plot the currents
    plt.figure()
    model.plot(model.i_rel(c, s, ip, r), color='b')
    model.plot(model.i_serca(c), color = 'r')
    model.plot(model.i_pmca(c), color = 'g')
    model.plot(model.i_leak(c, s), color = 'y')
    model.plot(model.i_out(c), color='k')
    plt.legend(['i_ip3r', 'i_serca', 'i_pmca', 'i_leak', 'i_out'])
    plt.show()

    

    
