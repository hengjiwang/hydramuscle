#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class DeYoungKeizerCell:
    '''Simplified De Young-Keizer model, modified from p101 of "Models of Calcium Signalling" by J. Sneyd'''
    def __init__(self, T = 100, dt = 0.001):
        # General Parameters
        self.delta = 1
        self.gamma = 5.5

        # IPR

        '''DeYoungKeizer'''
        self.v_ip3r = 0.5 # 1.11
        self.k_p = 0.13
        self.k_c = 0.082

        '''Hofer'''
        # self.v_ip3r = 0.08
        # self.ka = 0.2
        # self.kip = 0.3

        '''Sneyd'''
        # self.v_ip3r = 0.1
        # self.k_p = 0.3 # 0.01 (This value can make this model similar to Hofer model)
        # self.k_c = 0.7 # Atri
        # self.k_2 = 0.7 # Atri

        # Inactivation ratio
        self.ki = 0.2
        self.k_6 = 4

        # SERCA
        self.v_serca = 0.9
        self.k_serca = 0.1

        # PMCA
        self.v_pmca = 0.1
        self.k_pmca = 0.3

        # IP3
        self.ip_decay = 0.04

        # Initial values
        self.c0 = 0.05
        self.s0 = 200
        self.r0 = 0.94117
        self.ip0 = 0.01

        # Time constants
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))


    '''IP3R terms'''

    '''DeYoungKeizer'''
    def i_ip3r(self, c, s, r, ip):
            return self.v_ip3r * (ip / (self.k_p + ip))**3 \
                * (c / (self.k_c + c))**3 * r * (s - c)

    '''Hofer'''
    # def i_ip3r(self, c, s, r, ip):
    #     return (self.v_ip3r * r * c**2 * ip**2 / (self.ka**2 + c**2) / (self.kip**2 + ip**2)) * (s - c)

    '''Sneyd'''
    # def i_ip3r(self, c, c_t, r, ip):
    #     # Atri 1993 IP3R
    #     return self.v_ip3r * (ip**3 / (self.k_p**3 + ip**3)) \
    #         * (c / (self.k_c + c)) * r * ((c_t-c)*self.gamma - c)

    def v_r(self, c, r):
        # Rate of the inactivation ratio
        return self.k_6 * (self.ki**2 / (self.ki**2 + c**2) - r)

    '''SERCA'''
    def i_serca(self, c):
        # SERCA [uM/s]
        return self.v_serca * c**2 / (c**2 + self.k_serca**2)

    '''Leak from ER'''
    def i_leak(self, c, s):
        # Leak from ER to cytosol [uM/s]
        v_leak = (- self.i_ip3r(self.c0, self.s0, self.r0, self.ip0) 
        + self.i_serca(self.c0)) / (self.s0 - self.c0)
        return v_leak * (s - c)

    '''PMCA'''
    def i_pmca(self, c):
        return self.v_pmca * c**2 / (c**2 + self.k_pmca**2)

    '''Inward current'''
    def i_in(self):
        return self.i_pmca(self.c0)

    '''Stimulation'''
    def stim(self, t):
        if 20 <= t < 24:
            return 0.1
        else:
            return self.ip_decay * self.ip0

    '''Time stepping terms'''
    def rhs(self, y, t):
        # Right-hand side formulation
        c, s, r, ip = y

        dcdt = self.i_ip3r(c, s, r, ip) + self.i_leak(c, s) - self.i_serca(c) + (self.i_in() - self.i_pmca(c)) * self.delta
        dsdt = self.gamma * (self.i_serca(c) - self.i_ip3r(c, s, r, ip) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.stim(t) - self.ip_decay * ip

        return [dcdt, dsdt, drdt, dipdt]

    def step(self):
        # Time stepping    
        y0 = [self.c0, self.s0, self.r0, self.ip0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Plot'''
    def plot(self, a, tmin=0, tmax=1000, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":
    model = DeYoungKeizerCell(100)
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
    model.plot(model.i_ip3r(c, s, r, ip), color='b')
    model.plot(model.i_serca(c), color = 'r')
    model.plot(model.i_pmca(c), color = 'g')
    model.plot(model.i_leak(c, s), color = 'y')
    plt.legend(['i_ip3r', 'i_serca', 'i_pmca', 'i_leak'])
    plt.show()
    
