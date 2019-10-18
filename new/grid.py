#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from cell import Cell
import time
from fluo_encoder import FluoEncoder

class Grid(Cell, FluoEncoder):
    '''A 1D cell chain with cells connected through gap junctions'''
    def __init__(self, numx=20, numy=40, T=200, dt = 0.001, k2 = 0.3, s0 = 600, d = 20e-4, v7 = 0.02, k9 = 0.04):
        # Parameters
        FluoEncoder.__init__(self, T, dt)
        Cell.__init__(self, T, dt)
        self.gcx = 1000
        self.gcy = 1000
        self.g_ip3x = 0
        self.g_ip3y = 2
        self.numx = numx
        self.numy = numy
        onex = np.ones(self.numx)
        Ix = np.eye(numx)
        oney = np.ones(self.numy)
        Iy = np.eye(numy)
        Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.numx,self.numx).toarray()
        Dy = spdiags(np.array([oney,-2*oney,oney]),np.array([-1,0,1]),self.numy,self.numy).toarray()
        Dx[0, self.numx-1] = 1
        Dx[self.numx-1, 0] = 1
        Dy[0,0] = -1
        Dy[self.numy-1,self.numy-1] = -1 
        Dx = scipy.sparse.csr_matrix(Dx)
        Dy = scipy.sparse.csr_matrix(Dy)
        Ix = scipy.sparse.csr_matrix(Ix)
        Iy = scipy.sparse.csr_matrix(Iy)
        self.Lc = self.gcx * scipy.sparse.kron(Dx, Iy) + self.gcy * scipy.sparse.kron(Ix, Dy)
        self.Lip3 = self.g_ip3x * scipy.sparse.kron(Dx, Iy) + self.g_ip3y * scipy.sparse.kron(Ix, Dy)
        plt.figure()
        plt.imshow(self.Lip3.toarray(), 'rainbow')
        plt.colorbar()
        plt.show()
        self.k9 = k9
        self.d = d
        self.v7 = v7
        self.k2 = k2
        self.s0 = s0

        # Build grid
        self.num2 = self.numx * self.numy
        
        # General
        self.alpha = 1e9 / (2 * self.F * self.d)
        # self.beta = 5.5
    
    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation

        numx = self.numx
        numy = self.numy
        num2 = self.num2

        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = (y[0:num2], 
        y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2], y[4*num2:5*num2], 
        y[5*num2:6*num2], y[6*num2:7*num2], y[7*num2:8*num2], y[8*num2:9*num2], 
        y[9*num2:10*num2], y[10*num2:11*num2], y[11*num2:12*num2], y[12*num2:13*num2],
        y[13*num2:14*num2], y[14*num2:15*num2])

        iipr = self.i_ipr(c, s, ip, r)
        ileak = self.i_leak(c, s)
        iserca = self.i_serca(c)
        iin = self.i_in(ip)
        ipmca = self.i_pmca(c)
        ical = self.i_cal(v, n, hv, hc)
        icat = self.i_cat(v, bx, cx)
        vr = self.v_r(c, r)
        iplcb_base = self.i_plcb(self.v8)
        iplcb_stim = self.i_plcb(self.stim(t, stims_ip))    
        iplcd = self.i_plcd(c)
        ideg = self.i_deg(ip)
        ikca = self.i_kca(v, c)
        ibk = self.i_bk(v)
        istimv = self.stim_v(t, stims_v)
        ir1 = self.r_1(c, g, c1g)
        ir2 = self.r_2(c, c1g, c2g)
        ir3 = self.r_3(c, c2g, c3g)
        ir4 = self.r_4(c, c3g, c4g)

        dcdt =  (iipr +ileak - iserca) + (iin - ipmca - self.alpha * (ical + icat)) - ir1 - ir2 - ir3 - ir4
        dsdt = self.beta * (iserca - iipr - ileak)
        drdt = vr
        dipdt = iplcb_base + iplcd - ideg + self.Lip3.dot(ip)
        dipdt[(int(numx/2)-1)*numy] += iplcb_stim - iplcb_base
        dipdt[int(numx/2)*numy] += iplcb_stim - iplcb_base
        dipdt[(int(numx/2)+1)*numy] += iplcb_stim - iplcb_base
        dvdt = - 1 / self.c_m * (ical + icat + ikca + ibk) + self.Lc.dot(v)
        dvdt[0:3] += 1 / self.c_m * 0.01 * istimv
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)
        dgdt = - ir1
        dc1gdt = ir1 - ir2
        dc2gdt = ir2 - ir3
        dc3gdt = ir3 - ir4
        dc4gdt = ir4

        deriv = np.array([dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

        dydt = np.reshape(deriv, 15*num2)

        return dydt

    def step(self, stims_v = [201,203,205,207,209,211,213,215,217,219], stims_ip = [10]):
        # Time stepping
        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)
        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)

        base_mat = np.ones((self.numy, self.numx))
        inits = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.n0, self.hv0, 
                self.hc0, self.bx0, self.cx0, self.g0, self.c1g0, self.c2g0, self.c3g0, self.c4g0]
        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, 15*self.num2)  

        start_time = time.time() # Begin counting time
        sol = odeint(self.rhs, y0, self.time, args = (np.array(stims_v), np.array(stims_ip)), hmax=0.008)
        elapsed = (time.time() - start_time) # End counting time
        print("Num: " + str(self.numx) + ',' + str(self.numy) + "; Time used:" + str(elapsed))

        return sol

    def plot(self, a, tmin=0, tmax=300, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":
    model = Grid(numx=10, numy=20, T=100, dt=0.001)
    sol = model.step()
    df = pd.DataFrame(sol[:,0:model.numx*model.numy])
    df.to_csv('c_10x20_11s.csv', index = False)