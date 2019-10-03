#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/Users/hengjiwang/Documents/hydra_calcium_model/current/single/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
import pandas as pd
from cell import Cell

class Grid(Cell):
    '''A 2D dynamical model with cells connected by gap junctions'''
    def __init__(self, num = 10, T = 100):

        # General parameters
        super().__init__(T)
        self.gc = 5e4
        self.g_ip3 = 1

        # Build grid
        self.num = num
        onex = np.ones(self.num)
        Dx = spdiags(np.array([onex,-2*onex,onex]), np.array([-1,0,1]),self.num,self.num).toarray()
        Dx[0,0] = -1
        Dx[self.num-1,self.num-1] = -1 
        Ix = np.eye(self.num)
        self.L = np.kron(Dx, Ix) + np.kron(Ix, Dx)

        # Modified parameters
        self.ip_decay = 0.04

    '''Override stimulations'''
    def stim_v(self, t):
        # Voltage stimulation 
        if 1 <= t < 1.01 or 3 <= t < 3.01 or 5 <= t < 5.01:
            return 1
        else:
            return 0

    def stim(self, t):
        # IP3 stimulation
        if 25 <= t < 29:
            return 0.5
        else:
            return 0

    '''Time stepping'''
    def rhs(self, y, t):
        # Right-hand side formulation
        c, c_t, hh, ip, v, n, hv, hc, x, z, p, q = (y[0:self.num*self.num], 
        y[self.num*self.num:2*self.num*self.num], 
        y[2*self.num*self.num:3*self.num*self.num], 
        y[3*self.num*self.num:4*self.num*self.num], 
        y[4*self.num*self.num:5*self.num*self.num], 
        y[5*self.num*self.num:6*self.num*self.num], 
        y[6*self.num*self.num:7*self.num*self.num], 
        y[7*self.num*self.num:8*self.num*self.num], 
        y[8*self.num*self.num:9*self.num*self.num], 
        y[9*self.num*self.num:10*self.num*self.num], 
        y[10*self.num*self.num:11*self.num*self.num],
        y[11*self.num*self.num:12*self.num*self.num])

        # Cytosolic calcium
        dcdt = (self.i_ip3r(c, c_t, hh, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                + self.i_add(c, c_t)) * self.delta \
             - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)

        # Total calcium
        dctdt = (- self.i_pmca(c) + self.i_add(c, c_t)) * self.delta - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d)
        
        # Inactivation rate of IP3R
        dhhdt = (self.hh_inf(c, ip) - hh) / self.tau_hh(c, ip)

        # IP3 of downstream cells
        dipdt = (self.ip_decay * self.ip0 - self.ip_decay * ip) \
             + (self.i_plcd(c) - self.i_plcd(self.c0)) \
             + self.g_ip3 * self.L@ip
        
        # IP3 of stimulated cells
        dipdt[-int(self.num/2)-1              : -int(self.num/2) + 2             ] += self.stim(t)
        dipdt[-int(self.num/2)-1 -   self.num : -int(self.num/2) + 2 -   self.num] += self.stim(t)
        dipdt[-int(self.num/2)-1 - 2*self.num : -int(self.num/2) + 2 - 2*self.num] += self.stim(t)
    
        # Voltage of downstream cells
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) \
            + self.i_kcnq(v, x, z) + self.i_kv(v, p, q) \
            + self.i_bk(v)) \
            + self.gc * self.L@v
        
        # Voltage of stimulated cells
        dvdt[0:3*self.num] += 1 / self.c_m * 0.04 * self.stim_v(t)

        # CaL and K channel factors
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)
        dpdt = (self.p_inf(v) - p)/self.tau_p(v)
        dqdt = (self.q_inf(v) - q)/self.tau_q(v)
        
        # Put together
        deriv = np.array([dcdt, dctdt, dhhdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt, dpdt, dqdt])
        dydt = np.reshape(deriv, 12*self.num*self.num)        
        return dydt

    def step(self):
        # Time stepping
        self.hh0 = self.hh_inf(self.c0, self.ip0)

        y0 = np.array([self.c0*np.ones((self.num,self.num)), 
                       self.ct0*np.ones((self.num,self.num)), 
                       self.hh0*np.ones((self.num,self.num)), 
                       self.ip0*np.ones((self.num,self.num)), 
                       self.v0*np.ones((self.num,self.num)),
                       self.n0*np.ones((self.num,self.num)), 
                       self.hv0*np.ones((self.num,self.num)), 
                       self.hc0*np.ones((self.num,self.num)),
                       self.x0*np.ones((self.num,self.num)),
                       self.z0*np.ones((self.num,self.num)),
                       self.p0*np.ones((self.num,self.num)),
                       self.q0*np.ones((self.num,self.num))])

        y0 = np.reshape(y0, 12*self.num*self.num)   

        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

if __name__ == '__main__':
    n_cel = 5
    model = Grid(n_cel, 10)
    sol = model.step()
    # c = np.reshape(sol[:,0:n_cel*n_cel], (-1,n_cel,n_cel))
    # df = pd.DataFrame(np.reshape(c,(-1,n_cel**2)))
    df = pd.DataFrame(sol[:,0:n_cel*n_cel])
    df.to_csv('../save/data/c_5x5_10s.csv', index = False)


    

    
