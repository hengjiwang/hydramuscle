#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/hengji/Documents/hydra_calcium_model/single/')

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.sparse import spdiags
import pandas as pd
from cell import Cell
from numba import jitclass, int32, float32

@jitclass(spec)
class Grid(Cell):
    '''A 2D dynamical model with cells connected by gap junctions'''
    def __init__(self, num=10, T=100, k2=0.2, s0=600, d=10e-4, v7=0, k9=0.01, scale_stim_v = 0.01, scale_stim_ip = 1.0):

        # General parameters
        super().__init__(T)
        self.gc = 5e4
        self.g_ip3 = 1

        # Build grid
        self.num = num
        self.num2 = self.num ** 2
        onex = np.ones(self.num)
        Dx = spdiags(np.array([onex,-2*onex,onex]), np.array([-1,0,1]),self.num,self.num).toarray()
        Dx[0,0] = -1
        Dx[self.num-1,self.num-1] = -1 
        Ix = np.eye(self.num)
        self.L = np.kron(Dx, Ix) + np.kron(Ix, Dx)

        # Modified parameters
        # self.ip_decay = 0.04
        self.k9 = k9 
        self.d = d 
        self.v7 = v7 
        self.k2 = k2 
        self.s0 = s0
        self.scale_stim_v = scale_stim_v
        self.scale_stim_ip = scale_stim_ip

    '''Time stepping'''
    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation

        # Assign variables
        num = self.num
        num2 = self.num2
        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = (y[0:num2], 
        y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2], y[4*num2:5*num2], 
        y[5*num2:6*num2], y[6*num2:7*num2], y[7*num2:8*num2], y[8*num2:9*num2], 
        y[9*num2:10*num2], y[10*num2:11*num2], y[11*num2:12*num2], y[12*num2:13*num2],
        y[13*num2:14*num2], y[14*num2:15*num2])

        # Current terms which will be used for multiple times
        irel = self.i_rel(c, s, ip, r)
        ileak = self.i_leak(c, s)
        iserca = self.i_serca(c)
        iin = self.i_in(ip)
        ipmca = self.i_pmca(c)
        iout = self.i_out(c)
        ical = self.i_cal(v, n, hv, hc)
        icat = self.i_cat(v, bx, cx)
        iplcb_stim = self.scale_stim_ip * self.i_plcb(self.stim(t, stims_ip))
        iplcb_rest = self.i_plcb(self.v8)
        ir1 = self.r_1(c, g, c1g)
        ir2 = self.r_2(c, c1g, c2g)
        ir3 = self.r_3(c, c2g, c3g)
        ir4 = self.r_4(c, c3g, c4g)

        # Cytosolic calcium
        dcdt = irel + ileak - iserca + iin - ipmca - iout - 1e9 * (ical + icat) / (2 * self.F * self.d) \
            - ir1 - ir2 - ir3 - ir4

        # Total calcium
        dsdt = self.beta * (iserca - irel - ileak)
        
        # Inactivation rate of IP3R
        drdt = self.v_r(c, r)

        # IP3 of downstream cells
        dipdt = iplcb_rest + self.i_plcd(c) - self.i_deg(ip) \
             + self.g_ip3 * self.L@ip
        
        # IP3 of stimulated cells
        dipdt[-int(num/2)-1         : -int(num/2) + 2        ] += iplcb_stim - iplcb_rest
        dipdt[-int(num/2)-1 -   num : -int(num/2) + 2 -   num] += iplcb_stim - iplcb_rest
        dipdt[-int(num/2)-1 - 2*num : -int(num/2) + 2 - 2*num] += iplcb_stim - iplcb_rest
    
        # Voltage of downstream cells
        dvdt = - 1 / self.c_m * (ical + icat + self.i_kca(v, c) + self.i_bk(v)) \
            + self.gc * self.L@v
        
        # Voltage of stimulated cells
        dvdt[0:3*num] += 1 / self.c_m * self.scale_stim_v * self.stim_v(t, stims_v)

        # CaL and K channel factors
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)
        dgdt = - ir1
        dc1gdt = (ir1 - ir2)
        dc2gdt = (ir2 - ir3)
        dc3gdt = (ir3 - ir4)
        dc4gdt = ir4

        # Put together
        deriv = np.array([dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])
        dydt = np.reshape(deriv, 15*self.num2)        
        return dydt

    def step(self, stims_v = [201,203,205,207,209,211,213,215,217,219], stims_ip = [10]):
        # Time stepping

        start = time.clock() # Begin counting time

        self.r0 =  self.ki**2 / (self.ki**2 + self.c0**2)
        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)
        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)

        base_mat = np.ones((self.num,self.num))

        inits = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.n0, self.hv0, 
        self.hc0, self.bx0, self.cx0, self.g0, self.c1g0, self.c2g0, self.c3g0, self.c4g0]

        y0 = np.array([x*base_mat for x in inits])

        y0 = np.reshape(y0, 15*self.num2)   

        sol = odeint(self.rhs, y0, self.time, args = (stims_v, stims_ip), hmax = 0.005)

        elapsed = (time.clock() - start) # End counting time
        print("Time used:",elapsed)

        return sol

if __name__ == '__main__':
    n_cel = 5
    model = Grid(n_cel, 300)
    sol = model.step()
    # c = np.reshape(sol[:,0:n_cel*n_cel], (-1,n_cel,n_cel))
    # df = pd.DataFrame(np.reshape(c,(-1,n_cel**2)))
    df = pd.DataFrame(sol[:,0:n_cel*n_cel])
    df.to_csv('../save/data/c_5x5_300s.csv', index = False)


    

    
