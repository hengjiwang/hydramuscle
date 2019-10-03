#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from handy_cell import HandyCell

class AtriCell(HandyCell):
    '''Replaced the IP3R in HandyCell with that of Atri 1993'''
    def __init__(self, T = 20, dt = 0.001):
        # Inherit parameters
        super().__init__(T, dt)

        # New parameters
        self.k_p = 0.3 # 0.01 (This value can make this model similar to Hofer model)
        self.k_c = 0.7 # Atri
        self.k_2 = 0.7 # Atri
        self.tau_r = 0.2 # Sneyd
        self.k_out = 0 # 0.5 # Hofer

        # Modified parameters
        self.v_ip3r = 0.1 # 0.05 (Paired with k_p)
        self.r0 = self.k_2**2 / (self.k_2**2 + self.c0**2)
        self.ip_decay = 0.05

        # SERCA parameters (ModelsOfCalciumSignals, p101)
        self.v_serca = 0.9
        self.k_serca = 0.1
        
        # PMCA parameters (ModelsOfCalciumSignals, p101)
        self.v_pmca = 0.1
        self.k_pmca = 0.3
    
    def i_ip3r(self, c, c_t, r, ip):
        # Atri 1993 IP3R
        return self.v_ip3r * (ip**3 / (self.k_p**3 + ip**3)) \
            * (c / (self.k_c + c)) * r * ((c_t-c)*self.gamma - c)

    def i_serca(self, c):
        # SERCA [uM/s]
        return self.v_serca * c**2 / (c**2 + self.k_serca**2)

    def i_pmca(self, c):
        # PMCA [uM/s]
        return self.v_pmca * c**2 / (c**2 + self.k_pmca**2)

    def v_r(self, c, r):
        # Rate of the inactivation ratio
        return self.k_2**2 / (self.k_2**2 + c**2) - r

    def i_leak(self, c, c_t):
        # Leak from ER to cytosol [uM/s]
        v_leak = (- self.i_ip3r(self.c0, self.ct0, self.r0, self.ip0) 
        + self.i_serca(self.c0)) / ((self.ct0-self.c0)*self.gamma - self.c0)
        return v_leak * ((c_t-c)*self.gamma - c)

    def i_add(self, c, c_t):
        # Additional fluxes from the extracellular space [uM/s]
        v_in = self.k_out * self.c0 + self.i_pmca(self.c0)
        return v_in - self.k_out * c

    def stim(self, t):
        # Stimulation
        if 20 <= t < 40:
            return 0.05
        else:
            return self.ip_decay * self.ip0

    def rhs(self, y, t):
        # Right-hand side formulation
        c, c_t, r, ip = y

        dcdt = (self.i_ip3r(c, c_t, r, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                + self.i_add(c, c_t)) * self.delta
        dctdt = (- self.i_pmca(c) + self.i_add(c, c_t)) * self.delta
        drdt = self.v_r(c, r) / self.tau_r
        dipdt = self.stim(t) - self.ip_decay * ip

        return [dcdt, dctdt, drdt, dipdt]

    def step(self):
        # Time stepping
        y0 = [self.c0, self.ct0, self.r0, self.ip0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol


if __name__ == '__main__':
    model = AtriCell(T=100)
    sol = model.step()
    c = sol[:,0]
    c_t = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot((c_t - c) * model.gamma, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(r, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.show()

    # Plot the currents
    plt.figure()
    model.plot(model.i_ip3r(c, c_t, r, ip), color='b')
    model.plot(model.i_serca(c), color = 'r')
    model.plot(model.i_pmca(c), color = 'g')
    model.plot(model.i_leak(c, c_t), color = 'y')
    model.plot(model.i_add(c, c_t), color = 'k')
    plt.legend(['i_ip3r', 'i_serca', 'i_pmca', 'i_leak', 'i_add'])
    plt.show()
