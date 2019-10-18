#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../force/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from fast_cell import FastCell
from slow_cell import SlowCell
# from fluo_encoder import FluoEncoder
from maggio_force_encoder import MHMEncoder


class Cell(SlowCell, FastCell):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, T = 20, dt = 0.001, k2 = 0.05, s0 = 200, d = 10e-4, v7 = 0.04, v41 = 0.2):
        # Parameters
        SlowCell.__init__(self, T, dt)
        FastCell.__init__(self, T, dt)
        self.k2 = k2
        self.s0 = s0
        self.d = d
        self.v7 = v7
        self.v41 = v41

    def i_in(self, ip):
        return 1e9 * (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) + \
        self.i_cat(self.v0, self.bx0, self.cx0)) / (2 * self.F * self.d) + self.i_pmca(self.c0) + \
        self.v41 * ip**2 / (self.kr**2 + ip**2) - self.v41 * self.ip0**2 / (self.kr**2 + self.ip0**2)

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        g_bk = - (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) \
        + self.i_cat(self.v0, self.bx0, self.cx0) \
        + self.i_kca(self.v0, self.c0))/(self.v0 - self.e_bk)

        return g_bk * (v - self.e_bk)

    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation
        c, s, r, ip, v, n, hv, hc, bx, cx = y

        dcdt = self.i_ipr(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in(ip) - self.i_pmca(c)\
            - 1e9 * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx)) / (2 * self.F * self.d) 
        dsdt = self.beta * (self.i_serca(c) - self.i_ipr(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.stim(t, stims_ip)) + self.i_plcd(c) - self.i_deg(ip)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v) - 0.001 * self.stim_v(t, stims_v))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)

        return [dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt]

    def step(self, stims_v = [101,103,105,107,109,111,113,115,117,119], stims_ip = [10]):
        # Time stepping

        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)


        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.n0, self.hv0, 
        self.hc0, self.bx0, self.cx0]

        sol = odeint(self.rhs, y0, self.time, args = (stims_v, stims_ip, ), hmax = 0.005)

        return sol

    def plot(self, a, tmin=0, tmax=200, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = Cell(T=200, k2 = 0.01)
    sol = model.step()
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]
    v = sol[:,4]
    encoder = MHMEncoder(c)
    force = encoder.step()

    # Plot the results
    plt.figure()
    plt.subplot(241)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(242)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(243)
    model.plot(r, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(244)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.subplot(245)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(246)
    model.plot(force, ylabel = 'Active force')
    plt.show()