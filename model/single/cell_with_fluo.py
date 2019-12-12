#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../force/')
sys.path.insert(0, '../fluorescence/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from fast_cell import FastCell
from slow_cell import SlowCell
from maggio_force_encoder import MHMEncoder
from fluo_encoder import FluoEncoder
from tqdm import tqdm


class Cell(SlowCell, FastCell):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, fluo_buffer, T = 20, dt = 0.001, k2 = 0.05, s0 = 200, d = 40e-4, v7 = 0.03, v41 = 0.5):
        # Parameters
        SlowCell.__init__(self, T, dt)
        FastCell.__init__(self, T, dt)
        self.k2 = k2
        self.s0 = s0
        self.d = d
        self.v7 = v7
        self.v41 = v41
        self.fluo_buffer=fluo_buffer

    def i_in(self, ip):
        return 1e9 * (self.i_cal(self.v0, self.m0, self.h0) + \
        self.i_cat(self.v0, self.bx0, self.cx0)) / (2 * self.F * self.d) + self.i_pmca(self.c0) + \
        self.v41 * ip**2 / (self.kr**2 + ip**2) - self.v41 * self.ip0**2 / (self.kr**2 + self.ip0**2)

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        g_bk = - (self.i_cal(self.v0, self.m0, self.h0) \
        + self.i_cat(self.v0, self.bx0, self.cx0) \
        + self.i_kca(self.v0, self.c0))/(self.v0 - self.e_bk)

        return g_bk * (v - self.e_bk)

    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation
        c, s, r, ip, v, m, h, bx, cx, g, c1g, c2g, c3g, c4g = y

        ir1 = self.fluo_buffer.r_1(c, g, c1g)
        ir2 = self.fluo_buffer.r_2(c, c1g, c2g)
        ir3 = self.fluo_buffer.r_3(c, c2g, c3g)
        ir4 = self.fluo_buffer.r_4(c, c3g, c4g)

        dcdt = self.i_ipr(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in(ip) - self.i_pmca(c)\
            - 1e9 * (self.i_cal(v, m, h) + self.i_cat(v, bx, cx)) / (2 * self.F * self.d) - ir1 - ir2 - ir3 - ir4
        dsdt = self.beta * (self.i_serca(c) - self.i_ipr(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.stim(t, stims_ip)) + self.i_plcd(c) - self.i_deg(ip)
        dvdt = - 1 / self.c_m * (self.i_cal(v, m, h) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v) - 0.001 * self.stim_v(t, stims_v))
        dmdt = (self.m_inf(v) - m)/self.tau_m(v)
        dhdt = (self.h_inf(v) - h)/self.tau_h(v)
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)
        dgdt = - ir1
        dc1gdt = ir1 - ir2
        dc2gdt = ir2 - ir3
        dc3gdt = ir3 - ir4
        dc4gdt = ir4

        return [dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt]

    def step(self, stims_v = [101,103,105,107,109,111,113,115,117,119], stims_ip = [10, 30, 50]):
        # Time stepping

        self.m0 = self.m_inf(self.v0)
        self.h0 = self.h_inf(self.v0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)


        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.m0, self.h0, self.bx0, self.cx0, 
        self.fluo_buffer.g0, self.fluo_buffer.c1g0, self.fluo_buffer.c2g0, self.fluo_buffer.c3g0, self.fluo_buffer.c4g0]

        # sol = odeint(self.rhs, y0, self.time, args = (stims_v, stims_ip, ), hmax = 0.005)
        y = y0
        T = self.T
        dt = 0.0002

        sol = np.zeros((int(T/dt)+1, len(y0)))

        for j in tqdm(np.arange(0, int(T/dt)+1)):
            t = j*dt
            dydt = self.rhs(y, t, stims_v, stims_ip)
            y += np.array(dydt) * dt
            sol[j, :] = y

        return sol

    def plot(self, a, tmin=0, tmax=1000, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = Cell(FluoEncoder(None), T=600, k2 = 0.01)
    sol = model.step(stims_v = [1,3,5,7,9,12,15,18,22,26,31,36,42], stims_ip = [200, 220, 240])
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]
    v = sol[:,4]
    g = sol[:,9]
    c1g = sol[:,10]
    c2g = sol[:,11]
    c3g = sol[:,12]
    c4g = sol[:,13]
    fluo = model.fluo_buffer.f_total(g, c1g, c2g, c3g, c4g)
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
    plt.subplot(247)
    model.plot(fluo, ylabel = 'Fluorescence')
    plt.show()