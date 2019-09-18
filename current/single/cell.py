#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/hengji/Documents/hydra_calcium_model/current/fluorescence/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from fast_cell import FastCell
from hofer_cell import HoferCell
from fluo_encoder import FluoEncoder


class Cell(HoferCell, FastCell, FluoEncoder):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, T = 20, dt = 0.001):
        # Parameters
        # SlowCell.__init__(self, T, dt)
        FluoEncoder.__init__(self, None, T, dt)
        FastCell.__init__(self, T, dt)
        HoferCell.__init__(self, T, dt)

    def i_out(self, c):
        # Additional eflux [uM/s]
        return self.k5 * c

    def i_in(self):
        return 1e9 * (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) + self.i_cat(self.v0, self.bx0, self.cx0)) / (2 * self.F * self.d) + self.i_out(self.c0) + self.i_pmca(self.c0)

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        g_bk = - (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) \
        + self.i_cat(self.v0, self.bx0, self.cx0) \
        + self.i_kca(self.v0, self.c0))/(self.v0 - self.e_bk)
        return g_bk * (v - self.e_bk)

    '''Stimulation'''
    def stim(self, t):
        # Stimulation
        if 20 <= t < 24:
            return 1
        else:
            return self.v8

    def stim_v(self, t):
        # Stimulation
        if 1 <= t < 1.01 or 5 <= t < 5.01 or 9 <= t < 9.01 \
            or 12 <= t < 12.01 or 15 <= t < 15.01 or 17 <= t < 17.01 \
            or 19 <= t < 19.01 \
            or 21 <= t < 21.01 or 23 <= t < 23.01 or 25 <= t < 25.01 \
            or 27 <= t < 27.01 or 30 <= t < 30.01 or 33 <= t < 33.01 or 36 <= t < 36.01 \
            or 40 <= t < 40.01 or 43 <= t < 43.01:

        # if 101 <= t < 101.01 or 103 <= t < 103.01 or 105 <= t < 105.01 \
        #     or 109 <= t < 109.01 or 113 <= t < 113.01 or 117 <= t < 117.01 or 121 <= t < 121.01 \
        #     or 125 <= t < 125.01 \
        #     or 130 <= t < 130.01 or 135 <= t < 135.01 or 140 <= t < 140.01 \
        #     or 145 <= t < 145.01 or 150 <= t < 150.01 or 155 <= t < 155.01 or 160 <= t < 160.01 \
        #     or 166 <= t < 166.01 or 172 <= t < 172.01:
            return 0
        else:
            return 0

    def rhs(self, y, t):
        # Right-hand side formulation
        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = y

        dcdt = self.i_rel(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in() - self.i_pmca(c) - self.i_out(c)\
            - 1e9 * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx)) / (2 * self.F * self.d) - self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g) \
            - self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g)
        dsdt = self.beta * (self.i_serca(c) - self.i_rel(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.stim(t)) + self.i_plcd(c) - self.i_deg(ip)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v) - 0.001 * self.stim_v(t))
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

        return [dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt]

    def step(self):
        # Time stepping
        # self.hh0 = self.hh_inf(self.c0, self.ip0)
        
        self.r0 =  self.ki**2 / (self.ki**2 + self.c0**2)

        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)

        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.n0, self.hv0, 
        self.hc0, self.bx0, self.cx0, self.g0, self.c1g0, self.c2g0, self.c3g0, self.c4g0]

        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=100, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = Cell(T=100)
    sol = model.step()
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
    model.plot(fluo, ylabel = 'Fluorescence')
    plt.subplot(236)
    model.plot(v, ylabel = 'v[mV]')
    plt.show()