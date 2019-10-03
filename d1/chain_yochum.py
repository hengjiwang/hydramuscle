#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/hengji/Documents/hydra_calcium_model/current/single/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from yochum_cell import YochumCell

class Chain(YochumCell):
    '''A 1D cell chain with cells connected through gap junctions'''
    def __init__(self, num=20, T=20):
        # Parameters
        super().__init__(T)
        self.gc = 1000 # 5e4
        self.num = num
        onex = np.ones(self.num)
        self.Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.num,self.num).toarray()
        self.Dx[0,0] = -1
        self.Dx[self.num-1,self.num-1] = -1 
    
    def rhs(self, y, t):
        # Right-hand side formulation
        num = self.num

        c, v, n = (y[0:num], y[num:2*num], y[2*num:3*num])

        dcdt = self.f_c * (-self.alpha*self.i_ca(v, c) - self.k_ca * c)
        dvdt = 1/self.c_m * ( - self.i_ca(v, c) - self.i_k(v, n) - self.i_kca(v, c) - self.i_l(v)) + self.gc * self.Dx@v
        dvdt[0:3] += 1/self.c_m * 10 *  self.i_stim(t)
        dndt = (self.n_inf(v) - n) / self.tau_n(v)

        deriv = np.array([dcdt, dvdt, dndt])

        dydt = np.reshape(deriv, 3*num)

        return dydt

    def step(self):
        # Time stepping

        self.n0 = self.n_inf(self.v0)
        self.k_ca = -self.alpha*self.i_ca(self.v0, self.c0) / self.c0

        y0 = np.array([[self.c0]*self.num, 
                       [self.v0]*self.num, 
                       [self.n0]*self.num])

        y0 = np.reshape(y0, 3*self.num)
        
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=20, xlabel = 'time[s]', ylabel = None):
        # Plot function
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":

    n_cel = 20

    model = Chain(n_cel, 20)
    sol = model.step()
    c = sol[:,0:n_cel]
    v = sol[:,n_cel:2*n_cel]
    n = sol[:,2*n_cel:3*n_cel]

    # Plot the results
    plt.figure()
    plt.subplot(211)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(212)
    model.plot(v, ylabel = 'v[mV]')
    plt.show()

    



