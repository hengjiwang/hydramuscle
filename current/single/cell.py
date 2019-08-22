#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/Users/hengjiwang/Documents/hydra_calcium_model/current/fluorescence/')
sys.path.insert(0, '/Users/hengjiwang/Documents/hydra_calcium_model/current/force')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from fast_cell import FastCell
from hofer_cell import HoferCell
from fluo_encoder import FluoEncoder
from force_encoder import KatoForceEncoder


class Cell(HoferCell, FastCell):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, T = 100, dt = 0.001):
        # Parameters
        # SlowCell.__init__(self, T, dt)
        FastCell.__init__(self, T, dt)
        HoferCell.__init__(self, T, dt)
        self.d = 20e-4

    def i_out(self, c):
        # Additional eflux [uM/s]
        return self.k5 * c

    def i_in(self):
        return 1e9 * self.i_cal(self.v0, self.n0, self.hv0, self.hc0) / (2 * self.F * self.d) + self.i_out(self.c0) + self.i_pmca(self.c0)

    '''Stimulation'''
    def stim(self, t):
        # Stimulation
        if 20 <= t < 24:
            return 1
        else:
            return self.v8

    def stim_v(self, t):
        # Stimulation
        if 1 <= t < 1.01 or 3 <= t < 3.01 or 5 <= t < 5.01:
            return 1
        else:
            return 0

    def rhs(self, y, t):
        # Right-hand side formulation
        c, s, r, ip, v, n, hv, hc, x, z, p, q = y

        dcdt = self.i_rel(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in() - self.i_pmca(c) - self.i_out(c)\
            - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)
        dsdt = self.beta * (self.i_serca(c) - self.i_rel(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.stim(t)) + self.i_plcd(c) - self.i_deg(ip)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_kcnq(v, x, z) + self.i_kv(v, p, q) + self.i_bk(v) - 0.004 * self.stim_v(t))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)
        dpdt = (self.p_inf(v) - p)/self.tau_p(v)
        dqdt = (self.q_inf(v) - q)/self.tau_q(v)

        return [dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt, dpdt, dqdt]

    def step(self):
        # Time stepping
        # self.hh0 = self.hh_inf(self.c0, self.ip0)
        
        self.r0 =  self.ki**2 / (self.ki**2 + self.c0**2)
        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.n0, self.hv0, self.hc0, self.x0, self.z0, self.p0, self.q0]
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

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]', color = 'r--')
    plt.show()

    # Encode Fluorescence
    encoder = FluoEncoder(c)
    fluo = encoder.step()
    plt.figure(figsize = (15, 4))
    plt.subplot(211)
    model.plot(fluo, ylabel = 'Fluorescence', color='g')

    # Encode Force
    plt.subplot(212)
    force_encoder = KatoForceEncoder(c/1e6)
    force = force_encoder.step()
    model.plot(force, ylabel='Active Force', color='k')
    plt.show()


