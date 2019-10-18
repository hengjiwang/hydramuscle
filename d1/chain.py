#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/hengji/Documents/hydra_calcium_model/single/')

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
    def __init__(self, num=20, T=200, k2 = 0.2, s0 = 600, d = 10e-4, v7 = 0.04, k9 = 0.01):
        # Parameters
        super().__init__(T)
        self.gc = 1000 # 5e4
        self.g_ip3 = 2
        self.num = num
        onex = np.ones(self.num)
        self.Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.num,self.num).toarray()
        self.Dx[0,0] = -1
        self.Dx[self.num-1,self.num-1] = -1 
        self.k9 = k9 # 0.02
        self.d = d # 20e-4
        self.v7 = v7 # 0.06
        self.k2 = k2 # 0.05
        self.s0 = s0
    
    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation
        num = self.num

        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = (y[0:num], y[num:2*num], 
        y[2*num:3*num], y[3*num:4*num], y[4*num:5*num], 
        y[5*num:6*num], y[6*num:7*num], y[7*num:8*num], 
        y[8*num:9*num], y[9*num:10*num], y[10*num:11*num], y[11*num:12*num], y[12*num:13*num], 
        y[13*num:14*num], y[14*num:15*num])

        dcdt = self.i_rel(c, s, ip, r) + self.i_leak(c, s) - self.i_serca(c) + self.i_in(ip) - self.i_pmca(c) - self.i_out(c)\
            - 1e9 * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx)) / (2 * self.F * self.d) - self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g) \
            - self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g)
        dsdt = self.beta * (self.i_serca(c) - self.i_rel(c, s, ip, r) - self.i_leak(c, s))
        drdt = self.v_r(c, r)
        dipdt = self.i_plcb(self.v8) + self.i_plcd(c) - self.i_deg(ip) + self.g_ip3 * self.Dx@ip
        dipdt[0:3] += self.i_plcb(self.stim(t, stims_ip)) - self.i_plcb(self.v8)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v)) + self.gc * self.Dx@v
        dvdt[0:3] += 1 / self.c_m * 0.01 * self.stim_v(t, stims_v)
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

    def step(self, stims_v = [201,203,205,207,209,211,213,215,217,219], stims_ip = [10]):
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
        
        sol = odeint(self.rhs, y0, self.time, args = (stims_v, stims_ip), hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=300, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":

    n_cel = 40

    model = Chain(n_cel, 300)
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
    # df.to_csv('../save/data/c_40x1_300s_no_plcd_gip3_2.csv', index = False)

    



