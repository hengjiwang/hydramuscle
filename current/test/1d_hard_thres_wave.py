#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from hard_thres_wave import HardCell

class Chain(HardCell):
    '''A 1D cell chain with cells connected through gap junctions'''
    def __init__(self, num=20, T=20):
        # Parameters
        super().__init__(T)
        self.g_ip3 = 1
        self.num = num
        onex = np.ones(self.num)
        self.Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.num,self.num).toarray()
        self.Dx[0,0] = -1
        self.Dx[self.num-1,self.num-1] = -1 
        self.ip_decay = 0.1
    
    def rhs(self, y, t):
        # Right-hand side formulation
        num = self.num

        c, ip, r = y[0:num], y[num:2*num], y[2*num:3*num]

        dcdt = self.i_ip3r(ip, r) - self.i_ca_deg(c)
        drdt = self.v_r(c, r)
        dipdt = self.i_ip_deg(self.ip0) - self.i_ip_deg(ip) + self.g_ip3 * self.Dx@ip + 0.01 * c**2 / (c**2 + 0.3**2) - 0.01 * self.c0**2 / (self.c0**2 + 0.3**2)
        dipdt[0:3] += self.stim(t)

        deriv = [dcdt, dipdt, drdt]

        dydt = np.reshape(deriv, 3*num)

        return dydt

    def step(self):
        # Time stepping
        y0 = np.array([[self.c0]*self.num, [self.ip0]*self.num, [self.r0]*self.num])
        y0 = np.reshape(y0, 3*self.num)
        
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    def plot(self, a, tmin=0, tmax=1000, xlabel = 'time[s]', ylabel = None):
        # Plot function
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == "__main__":

    n_cel = 50

    model = Chain(n_cel, 100)
    sol = model.step()
    c = sol[:,0:n_cel]
    ip = sol[:,n_cel:2*n_cel]

    # Plot the results
    plt.figure()
    plt.subplot(121)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(122)
    model.plot(ip, ylabel = 'ip3[uM]')
    plt.show()