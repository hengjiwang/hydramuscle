#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/hengji/Documents/hydra_calcium_model/current/single/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
# from young_keizer_cell import DeYoungKeizerCell
# from fast_cell import FastCell
from cell import Cell

class Chain(Cell):
    '''A 1D cell chain with cells connected through gap junctions'''
    def __init__(self, num=20, T=20):
        # Parameters
        super().__init__(T)
        self.gc = 1000 # 5e4
        self.g_ip3 = 1
        self.num = num
        onex = np.ones(self.num)
        self.Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.num,self.num).toarray()
        self.Dx[0,0] = -1
        self.Dx[self.num-1,self.num-1] = -1 
        self.k9 = 0.02
        # self.ki = 0.5
        # self.s0 = 100
        self.d = 20e-4
        # self.k3 = 1
        self.v7 = 0.06
        self.k2 = 0.05
        self.s0 = 200

    def stim(self, t):
        # Stimulation
        if 20 <= t < 24:
            return 1
        else:
            return self.v8

    def stim_v(self, t):
        # Stimulation

        # if 1 <= t < 1.01 or 5 <= t < 5.01 or 9 <= t < 9.01 \
        #     or 12 <= t < 12.01 or 15 <= t < 15.01 or 17 <= t < 17.01 \
        #     or 19 <= t < 19.01 \
        #     or 21 <= t < 21.01 or 23 <= t < 23.01 or 25 <= t < 25.01 \
        #     or 27 <= t < 27.01 or 30 <= t < 30.01 or 33 <= t < 33.01 or 36 <= t < 36.01 \
        #     or 40 <= t < 40.01 or 43 <= t < 43.01:

        if 101 <= t < 101.01 or 103 <= t < 103.01 or 105 <= t < 105.01 \
            or 109 <= t < 109.01 or 113 <= t < 113.01 or 117 <= t < 117.01 or 121 <= t < 121.01 \
            or 125 <= t < 125.01 \
            or 130 <= t < 130.01 or 135 <= t < 135.01 or 140 <= t < 140.01 \
            or 145 <= t < 145.01 or 150 <= t < 150.01 or 155 <= t < 155.01 or 160 <= t < 160.01 \
            or 166 <= t < 166.01 or 172 <= t < 172.01:
            return 1
        else:
            return 0
    
    def rhs(self, y, t):
        # Right-hand side formulation
        num = self.num

        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = (y[0:num], y[num:2*num], 
        y[2*num:3*num], y[3*num:4*num], y[4*num:5*num], 
        y[5*num:6*num], y[6*num:7*num], y[7*num:8*num], 
        y[8*num:9*num], y[9*num:10*num], y[10*num:11*num], y[11*num:12*num], y[12*num:13*num], 
        y[13*num:14*num], y[14*num:15*num])

        dcdt = self.i_rel(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in() - self.i_pmca(c) - self.i_out(c)\
            - 1e9 * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx)) / (2 * self.F * self.d) - self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g) \
            - self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g)
        dsdt = self.beta * (self.i_serca(c) - self.i_rel(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.v8) + self.i_plcd(c) - self.i_deg(ip) + self.g_ip3 * self.Dx@ip
        dipdt[0:3] += self.i_plcb(self.stim(t)) - self.i_plcb(self.v8)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v)) + self.gc * self.Dx@v
        dvdt[0:3] += 1 / self.c_m * 0.01 * self.stim_v(t)
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

        deriv = np.array([dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

        dydt = np.reshape(deriv, 15*num)

        return dydt

    def step(self):
        # Time stepping

        self.r0 =  self.ki**2 / (self.ki**2 + self.c0**2)
        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)
        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)

        y0 = np.array([[self.c0]*self.num, 
                       [self.s0]*self.num, 
                       [self.r0]*self.num, 
                       [self.ip0]*self.num,
                       [self.v0]*self.num,
                       [self.n0]*self.num,
                       [self.hv0]*self.num,
                       [self.hc0]*self.num,
                       [self.bx0]*self.num,
                       [self.cx0]*self.num,
                       [self.g0]*self.num,
                       [self.c1g0]*self.num,
                       [self.c2g0]*self.num,
                       [self.c3g0]*self.num,
                       [self.c4g0]*self.num])

        y0 = np.reshape(y0, 15*self.num)
        
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=200, xlabel = 'time[s]', ylabel = None):
        # Plot function
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":

    n_cel = 20

    model = Chain(n_cel, 200)
    sol = model.step()
    c = sol[:,0:n_cel]
    s = sol[:,n_cel:2*n_cel]
    r = sol[:,2*n_cel:3*n_cel]
    ip = sol[:,3*n_cel:4*n_cel]
    v = sol[:, 4*n_cel:5*n_cel]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.show()

    # Save the [Ca2+]
    df = pd.DataFrame(sol[:,0:n_cel])
    df.to_csv('../save/data/c_20x1_200s_include_fluo.csv', index = False)

    



