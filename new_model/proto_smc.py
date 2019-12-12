#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from fast_cell import FastCell
from slow_cell import SlowCell

class ProtoSMC(SlowCell, FastCell):

    def __init__(self, T = 20, dt = 0.001, k2 = 0.05, s0 = 60, d = 20e-4, v7 = 0.04):
        SlowCell.__init__(self, T, dt)
        FastCell.__init__(self, T, dt)

        self.k2 = k2
        self.s0 = s0
        self.d = d
        self.v7 = v7
        self.alpha = 1e9 / (2 * self.F * self.d)

    '''Overload methods'''
    def i_in(self, ip):
        return self.alpha * (self.ical0 + self.icat0) + self.ipmca0 + self.v41 * ip**2 / (self.kr**2 + ip**2) - self.in_ip0

    # def i_bk(self, v):
    #     # Background voltage leak [mA/cm^2]
    #     return self.g_bk * (v - self.e_bk)

    '''Numerical calculations'''
    def rhs(self, y, t, stims_fast, stims_slow):
        # Right-hand side formulation
        c, s, r, ip, v, m, h, bx, cx = y

        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.calc_slow_terms(c, s, r, ip)
        _, i_cal, i_cat, i_kca, i_bk, dmdt, dhdt, dbxdt, dcxdt = self.calc_fast_terms(c, v, m, h, bx, cx)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.alpha * (i_cal + i_cat)
        dsdt = self.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow)) + i_plcd - i_deg

        dvdt = - 1 / self.c_m * (i_cal + i_cat + i_kca + i_bk - 0.001 * self.stim_fast(t, stims_fast))

        return np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dbxdt, dcxdt])


    def run(self, stims_fast, stims_slow):
        # Run the model
        self.init_fast_cell()
        self.init_slow_cell()

        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.m0, self.h0, self.bx0, self.cx0]

        # y = y0
        # T = self.T
        # dt = self.dt

        sol = self.euler_odeint(self.rhs, y0, self.T, self.dt, stims_fast=stims_fast, stims_slow=stims_slow)

        return sol


if __name__ == '__main__':
    model = ProtoSMC(T=200, dt = 0.0002, k2 = 0.01)
    sol = model.run(stims_fast = [1,3,5,7,9,12,15,18,22,26,31,36,42], stims_slow = [100])
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]
    v = sol[:,4]

    # Plot the results
    plt.figure()
    plt.subplot(231)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(232)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(233)
    model.plot(r, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(234)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.subplot(235)
    model.plot(v, ylabel = 'v[mV]')
    plt.show()