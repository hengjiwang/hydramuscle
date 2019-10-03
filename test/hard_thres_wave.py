#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HardCell:
    # Simple Model for testing whether a hard-threshold on IP3R 
    # can give rise to a wave-form propagation
    def __init__(self, T = 100, dt = 0.001):

        # Coefficients
        self.ca_deg = 5
        self.ip_deg = 0.04

        self.k6 = 0.1
        self.ki = 0.2

        # Initial values
        self.c0 = 0.05
        self.ip0 = 0.01
        self.r0 = 0.94

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))


    '''IP3R'''
    def i_ip3r(self, ip, r):
        # return ((ip > 0.0101)*r + self.i_ca_deg(self.c0))
        return ip**10 / (ip**10 + 0.02**10) * r + self.i_ca_deg(self.c0) - self.ip0**10 / (self.ip0**10 + 0.02**10)*self.r0

    def v_r(self, c, r):
        return self.k6 * (self.ki**2 / (self.ki**2 + c**2) - r)

    '''Stimulation'''
    def stim(self, t):
        if 1 <= t < 5:
            return 0.1
        else:
            return self.i_ip_deg(self.ip0)

    '''Degrations'''

    def i_ca_deg(self, c):
        return self.ca_deg * c**2 / (0.7**2 + c**2)

    def i_ip_deg(self, ip):
        return self.ip_deg * ip

    '''Time stepping'''
    def rhs(self, y, t):
        c, ip, r = y

        dcdt = self.i_ip3r(ip, r) - self.i_ca_deg(c)
        drdt = self.v_r(c, r)
        dipdt = self.stim(t) - self.i_ip_deg(ip) + 0.02 * c**2 / (c**2 + 0.1**2) - 0.02 * self.c0**2 / (self.c0**2 + 0.1**2)


        return [dcdt, dipdt, drdt]

    def step(self):
        y0 = [self.c0, self.ip0, self.r0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Plot'''
    def plot(self, a, tmin=0, tmax=100, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)


if __name__ == "__main__":

    model = HardCell(T=100)
    sol = model.step()
    c = sol[:,0]
    r = sol[:,2]
    ip = sol[:,1]

    plt.figure()
    plt.subplot(311)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(312)
    model.plot(ip, ylabel = 'ip3[uM]')
    plt.subplot(313)
    model.plot(r, ylabel = 'r')

    plt.show()

    