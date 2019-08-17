#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from slow_cell import SlowCell


class Cell(SlowCell):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, T = 20, dt = 0.001):
        # Parameters
        super().__init__(T, dt)

    def i_add(self, c, c_t):
        # Additional fluxes from the extracellular space [uM/s]
        k_out = (self.v_in - self.i_pmca(self.c0)) / self.c0
        return self.v_in - k_out * c

    def stim(self, t):
        # Stimulation
        if 2 <= t < 2.1 or 3 <= t < 3.1 or 4 <= t < 4.1 or 5 <= t < 5.1:
            return 1
        else:
            return 0 # self.ip_decay * self.ip0

    def stim2(self, t):
        # Stimulation
        if None:
            return 0.5
        else:
            return self.ip_decay * self.ip0

    def rhs(self, y, t):
        # Right-hand side formulation
        c, c_t, hh, ip = y

        dcdt = (self.i_ip3r(c, c_t, hh, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                + self.i_add(c, c_t) + 40 * self.stim(t)) * self.delta

        dctdt = (- self.i_pmca(c) + self.i_add(c, c_t) + 40 * self.stim(t)) * self.delta
        dhhdt = (self.hh_inf(c, ip) - hh) / self.tau_hh(c, ip)
        dipdt = self.stim2(t) - self.ip_decay * ip
        # dipdt = 0

        return [dcdt, dctdt, dhhdt, dipdt]

    def step(self):
        # Time stepping
        self.hh0 = self.hh_inf(self.c0, self.ip0)
        
        y0 = [self.c0, self.ct0, self.hh0, self.ip0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=10, xlabel = 'time[s]', ylabel = None):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = Cell(T=10)
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

