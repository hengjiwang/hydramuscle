#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/hengji/Documents/hydra_calcium_model/current/fluorescence/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from handy_cell import HandyCell
from fast_cell import FastCell
from fluo_encoder import FluoEncoder



class Cell(HandyCell, FastCell):
    # This is a intracellular model without L-type calcium channel
    def __init__(self, T = 20, dt = 0.001):
        # Parameters
        HandyCell.__init__(self, T, dt)
        FastCell.__init__(self, T, dt)

        self.v_plcd = 0.03 # 0.06
        self.k_plcd = 0.3
        # self.v_ip3r = 0.1

    def i_add(self, c, c_t):
        # Additional fluxes from the extracellular space [uM/s]
        k_out = (self.v_in - self.i_pmca(self.c0) - 1e9 * self.i_cal(self.v0, self.n0, self.hv0, self.hc0) / (2 * self.F * self.d) / self.delta ) / self.c0
        return self.v_in - k_out * c

    def i_plcd(self, c):
        return self.v_plcd * c**2 / (c**2 + self.k_plcd**2)

    def stim_ip(self, t):
        # Stimulation
        if 70 <= t < 80:
            return 0.04 * 1.5
        else:
            return self.ip_decay * self.ip0 - self.i_plcd(self.c0)

    def stim_v(self, t):
        # Stimulation
        if 1 <= t < 1.01 or 3 <= t < 3.0 or 5 <= t < 5.01 or 6.5 <= t < 6.51 \
            or 8 <= t < 8.01 or 9.5 <= t < 9.51 or 11 <= t < 11.01 or 12.5 <= t < 12.51 \
            or 14 <= t < 14.01 or 15.5 <= t < 15.51 \
            or 17 <= t < 17.01 or 19.5 <= t < 19.51:
            # or 15.6 <= t < 15.61 or 17 <= t < 17.01 or 18.5 <= t < 18.51 or 20.1 <= t < 20.11 or 21.8 <= t < 21.81:
            # or 39 <= t < 39.01 or 41 <= t < 41.01 or 43 <= t < 43.01 or 45 <= t < 45.01:
            return 1
        else:
            return 0

    def rhs(self, y, t):
        # Right-hand side formulation

        self.hh0 = self.hh_inf(self.c0, self.ip0)

        c, c_t, hh, ip, v, n, hv, hc, x, z, p, q = y

        dcdt = (self.i_ip3r(c, c_t, hh, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                + self.i_add(c, c_t)) * self.delta \
             - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)

        dctdt = (- self.i_pmca(c) + self.i_add(c, c_t)) * self.delta - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)
        dhhdt = (self.hh_inf(c, ip) - hh) / self.tau_hh(c, ip)
        dipdt = self.stim_ip(t) - self.ip_decay * ip + self.i_plcd(c)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_kcnq(v, x, z) + self.i_kv(v, p, q) + self.i_bk(v) - 0.004 * self.stim_v(t))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)
        dpdt = (self.p_inf(v) - p)/self.tau_p(v)
        dqdt = (self.q_inf(v) - q)/self.tau_q(v)

        return [dcdt, dctdt, dhhdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt, dpdt, dqdt]

    def step(self):
        # Time stepping
        self.hh0 = self.hh_inf(self.c0, self.ip0)
        
        y0 = [self.c0, self.ct0, self.hh0, self.ip0, self.v0, self.n0, self.hv0, self.hc0, self.x0, self.z0, self.p0, self.q0]
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
    c_t = sol[:,1]
    hh = sol[:,2]
    ip = sol[:,3]
    v = sol[:,4]

    # Encode to fluorescence

    fluo_encoder = FluoEncoder(c, model.T)
    fluo = fluo_encoder.step()

    # Plot the results
    plt.figure()
    plt.subplot(231)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(232)
    model.plot((c_t - c) * model.gamma, ylabel = 'c_ER[uM]')
    plt.subplot(233)
    model.plot(hh, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(234)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.subplot(235)
    model.plot(fluo, ylabel = 'Fluorescence')
    plt.subplot(236)
    model.plot(v, ylabel = 'v[mV]')
    plt.show()