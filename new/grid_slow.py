#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from slow_cell import SlowCell
import time
from fluo_encoder import FluoEncoder

class Grid(SlowCell, FluoEncoder):
    '''A 1D cell chain with cells connected through gap junctions'''
    def __init__(self, numx=20, numy=40, T=200, dt = 0.001, k2 = 0.2, s0 = 600, v7 = 0.03, k9 = 0.06):
        # Parameters
        FluoEncoder.__init__(self, T, dt)
        SlowCell.__init__(self, T, dt)
        self.g_ip3x = 0
        self.g_ip3y = 1
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
        self.Lip3 = self.g_ip3x * scipy.sparse.kron(Dx, Iy) + self.g_ip3y * scipy.sparse.kron(Ix, Dy)
        # self.Lip3 = self.g_ip3x * np.kron(Dx, Iy) + self.g_ip3y * np.kron(Ix, Dy)
        plt.figure()
        plt.imshow(self.Lip3.toarray(), 'rainbow')
        plt.colorbar()
        plt.show()
        self.k9 = k9
        self.v7 = v7
        self.k2 = k2
        self.s0 = s0

        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

        # Build grid
        self.num2 = self.numx * self.numy
    
    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation

        numx = self.numx
        numy = self.numy
        num2 = self.num2

        c, s, r, ip, g, c1g, c2g, c3g, c4g = (y[0:num2], 
        y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2], y[4*num2:5*num2], 
        y[5*num2:6*num2], y[6*num2:7*num2], y[7*num2:8*num2], y[8*num2:9*num2])

        iipr = self.i_ipr(c, s, ip, r)
        ileak = self.i_leak(c, s)
        iserca = self.i_serca(c)
        iin = self.i_in(ip)
        ipmca = self.i_pmca(c)
        vr = self.v_r(c, r)
        ideg = self.i_deg(ip)
        iplcb_base = self.i_plcb(self.v8)
        iplcb_stim = self.i_plcb(self.stim(t, stims_ip))    
        iplcd = self.i_plcd(c)
        ir1 = self.r_1(c, g, c1g)
        ir2 = self.r_2(c, c1g, c2g)
        ir3 = self.r_3(c, c2g, c3g)
        ir4 = self.r_4(c, c3g, c4g)

        dcdt =  (iipr +ileak - iserca) + (iin - ipmca) - ir1 - ir2 - ir3 - ir4
        dsdt = self.beta * (iserca - iipr - ileak)
        drdt = vr
        dipdt = iplcb_base + iplcd - ideg + self.Lip3.dot(ip)
        dipdt[(int(numx/2)-1)*numy] += iplcb_stim - iplcb_base
        dipdt[int(numx/2)*numy] += iplcb_stim - iplcb_base
        dipdt[(int(numx/2)+1)*numy] += iplcb_stim - iplcb_base
        dgdt = - ir1
        dc1gdt = ir1 - ir2
        dc2gdt = ir2 - ir3
        dc3gdt = ir3 - ir4
        dc4gdt = ir4

        deriv = np.array([dcdt, dsdt, drdt, dipdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

        dydt = np.reshape(deriv, 9*num2)

        return dydt

    def step(self, stims_v = [201,203,205,207,209,211,213,215,217,219], stims_ip = [10]):
        # Time stepping
        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)

        base_mat = np.ones((self.numy, self.numx))
        inits = [self.c0, self.s0, self.r0, self.ip0, self.g0, self.c1g0, self.c2g0, self.c3g0, self.c4g0]
        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, 9*self.num2)  

        start_time = time.time() # Begin counting time
        sol = odeint(self.rhs, y0, self.time, args = (np.array(stims_v), np.array(stims_ip)), hmax=0.008, atol=0.01, rtol=0.01)
        elapsed = (time.time() - start_time) # End counting time
        print("Num: " + str(self.numx) + ',' + str(self.numy) + "; Time used:" + str(elapsed))

        return sol

    def plot(self, a, tmin=0, tmax=300, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":
    model = Grid(numx=20, numy=20, T=100, dt=0.1)
    sol = model.step()
    df = pd.DataFrame(sol[:,0:model.numx*model.numy])
    df.to_csv('c_20x20_100s_slow.csv', index = False)