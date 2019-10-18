#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../fluorescence/')
sys.path.insert(0, '../force/')
sys.path.insert(0, '../single/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from fast_cell import FastCell
from hofer_cell import HoferCell
from fluo_encoder import FluoEncoder
from maggio_force_encoder import MHMEncoder


class Cell(HoferCell, FastCell, FluoEncoder):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, T = 20, dt = 0.001, k2 = 0.05, s0 = 200, d = 10e-4, v7 = 0.04, v41 = 0.2):
        # Parameters
        # SlowCell.__init__(self, T, dt)
        FluoEncoder.__init__(self, None, T, dt)
        FastCell.__init__(self, T, dt)
        HoferCell.__init__(self, T, dt)
        self.k2 = k2 # 0.1
        self.s0 = s0
        self.d = d # 20e-4
        self.v7 = v7 # 0.04
        self.v41 = v41

    def i_out(self, c):
        # Additional eflux [uM/s]
        return self.k5 * c

    def i_in(self, ip):
        return 1e9 * (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) + \
        self.i_cat(self.v0, self.bx0, self.cx0)) / (2 * self.F * self.d) + self.i_out(self.c0) + self.i_pmca(self.c0) + \
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
        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = y

        dcdt = self.i_rel(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in(ip) - self.i_pmca(c) - self.i_out(c) \
            - 1e9 * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx)) / (2 * self.F * self.d) \
            - self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g) - self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g) \
            + self.r_1(self.c0, self.g0, self.c1g0) + self.r_2(self.c0, self.c1g0, self.c2g0) + \
                self.r_3(self.c0, self.c2g0, self.c3g0) + self.r_4(self.c0, self.c3g0, self.c4g0)
        dsdt = self.beta * (self.i_serca(c) - self.i_rel(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.stim(t, stims_ip)) + self.i_plcd(c) - self.i_deg(ip)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v) - 0.001 * self.stim_v(t, stims_v))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)
        dgdt = - self.r_1(c, g, c1g)
        dc1gdt = (self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g))
        dc2gdt = (self.r_2(c, c1g, c2g) - self.r_3(c, c2g, c3g))
        dc3gdt = (self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g))
        dc4gdt = self.r_4(c, c3g, c4g)

        # if t < 0.001:   print(dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt)

        return [dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt]

    def step(self, stims_v = [101,103,105,107,109,111,113,115,117,119], stims_ip = [10]):
        # Time stepping
        
        self.r0 =  self.ki**2 / (self.ki**2 + self.c0**2)

        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)


        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.n0, self.hv0, 
        self.hc0, self.bx0, self.cx0, self.g0, self.c1g0, self.c2g0, self.c3g0, self.c4g0]

        sol = odeint(self.rhs, y0, self.time, args = (stims_v, stims_ip, ), hmax = 0.005)

        return sol

    def plot(self, a, tmin=0, tmax=200, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = Cell(T=200, k2 = 0.01)
    sol = model.step(stims_v = [-100], stims_ip = [-100])
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]
    v = sol[:,4]
    g = sol[:, 10]
    c1g = sol[:, 11]
    c2g = sol[:, 12]
    c3g = sol[:, 13]
    c4g = sol[:, 14]
    fluo = model.f_total(g, c1g, c2g, c3g, c4g)
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
    model.plot(fluo, ylabel = 'Fluorescence')
    plt.subplot(246)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(247)
    model.plot(force, ylabel = 'Active force')
    plt.show()