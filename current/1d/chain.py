#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/Users/hengjiwang/Documents/hydra_calcium_model/current/single/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from young_keizer_cell import DeYoungKeizerCell
from fast_cell import FastCell
from cell import Cell

class Chain(Cell):
    '''A 1D cell chain with cells connected through gap junctions'''
    def __init__(self, num=20, T=20):
        # Parameters
        super().__init__(T)
        self.gc = 5e4
        self.g_ip3 = 5
        self.num = num
        onex = np.ones(self.num)
        self.Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.num,self.num).toarray()
        self.Dx[0,0] = -1
        self.Dx[self.num-1,self.num-1] = -1 
        self.ip_decay = 0.02

    def stim_v(self, t):
        # Stimulation 
        if 1 <= t < 1.01 or 3 <= t < 3.01 or 5 <= t < 5.01:
            return 1
        else:
            return 0

    def stim(self, t):
        if 20 <= t < 24:
            return 0.5 # 0.1
        else:
            return self.ip_decay * self.ip0
    
    def rhs(self, y, t):
        # Right-hand side formulation
        num = self.num

        c, c_t, hh, ip, v, n, hv, hc, x, z, p, q = (y[0:num], y[num:2*num], y[2*num:3*num], y[3*num:4*num], y[4*num:5*num], 
        y[5*num:6*num], y[6*num:7*num], y[7*num:8*num], y[8*num:9*num], y[9*num:10*num], y[10*num:11*num], y[11*num:12*num])

        dcdt = (self.i_ip3r(c, c_t, hh, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                + self.i_add(c, c_t)) * self.delta \
             - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)

        dctdt = (- self.i_pmca(c) + self.i_add(c, c_t)) * self.delta - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)
        dhhdt = (self.hh_inf(c, ip) - hh) / self.tau_hh(c, ip)
        dipdt = self.ip_decay * self.ip0 - self.ip_decay * ip + self.g_ip3 * self.Dx@ip
        dipdt[0:6] += self.stim(t) - self.ip_decay * self.ip0
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_kcnq(v, x, z) + self.i_kv(v, p, q) + self.i_bk(v)) + self.gc*self.Dx@v
        dvdt[0:3] += 1 / self.c_m * 0.1 * self.stim_v(t)
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)
        dpdt = (self.p_inf(v) - p)/self.tau_p(v)
        dqdt = (self.q_inf(v) - q)/self.tau_q(v)

        deriv = np.array([dcdt, dctdt, dhhdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt, dpdt, dqdt])

        dydt = np.reshape(deriv, 12*num)

        return dydt

    def step(self):
        # Time stepping

        self.hh0 = self.hh_inf(self.c0, self.ip0)

        y0 = np.array([[self.c0]*self.num, 
                       [self.ct0]*self.num, 
                       [self.hh0]*self.num, 
                       [self.ip0]*self.num,
                       [self.v0]*self.num,
                       [self.n0]*self.num,
                       [self.hv0]*self.num,
                       [self.hc0]*self.num,
                       [self.x0]*self.num,
                       [self.z0]*self.num,
                       [self.p0]*self.num,
                       [self.q0]*self.num,])

        y0 = np.reshape(y0, 12*self.num)
        
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=100, xlabel = 'time[s]', ylabel = None):
        # Plot function
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":

    n_cel = 50

    model = Chain(n_cel, 100)
    sol = model.step()
    c = sol[:,0:n_cel]
    c_t = sol[:,n_cel:2*n_cel]
    hh = sol[:,2*n_cel:3*n_cel]
    ip = sol[:,3*n_cel:4*n_cel]
    v= sol[:, 4*n_cel:5*n_cel]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot((c_t - c) * model.gamma, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.show()

    



