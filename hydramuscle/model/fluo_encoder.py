#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class FluoEncoder:
    
    # Init
    def __init__(self, c, T=100, dt=0.001):
        '''
        :param c: [Ca2+] as input
        :param T: Total time
        :param dt: Numerical step length
        '''
        self.r_inc = 200
        self.tau_ex = 1.0
        self.k1p = 20 # 2.5
        self.k1n = 0.1
        self.k2p = 16.9
        self.k2n = 205
        self.k3p = 30 # 1.1
        self.k3n = 11.8
        self.k4p = 1069
        self.k4n = 2 # 3 # 5.8
        self.c = c
        self.g0 = 0.6614852922934815 # 3.28088468
        self.c1g0 = 6.614852922934705 # 4.10110585
        self.c2g0 = 0.027266101072584505 # 0.01690456
        self.c3g0 = 0.0034660297973623183 # 7.87924326e-05
        self.c4g0 = 0.09262964633450728 # 0.00072611
        self.phi0 = 1
        self.phi1 = 1
        self.phi2 = 1
        self.phi3 = 1
        self.phi4 = 81
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, self.T, int(self.T/self.dt))
        
    
    # Fluorescence
    def f_total(self, g, c1g, c2g, c3g, c4g):
        f_cyt = self.phi0*g + self.phi1*c1g + self.phi2*c2g + \
        self.phi3*c3g + self.phi4*c4g
        f_cyt0 = self.phi0*self.g0 + self.phi1*self.c1g0 + \
        self.phi2*self.c2g0 + self.phi3*self.c3g0 + self.phi4*self.c4g0
        f_bg = f_cyt0 * 2.5
        return (f_cyt + f_bg) / (f_cyt0 + f_bg)
    
    # rate 1
    def r_1(self, c, g, c1g):
        return self.k1p * c * g - self.k1n * c1g
    
    # rate 2
    def r_2(self, c, c1g, c2g):
        return self.k2p * c * c1g - self.k2n * c2g
    
    # rate_3
    def r_3(self, c, c2g, c3g):
        return self.k3p * c * c2g - self.k3n * c3g
    
    # rate_4
    def r_4(self, c, c3g, c4g):
        return self.k4p * c * c3g - self.k4n * c4g
    
    # Stimulation rate
    def stim(self, t):
        if t >= 10 and t < 10.01:
            return self.r_inc
        # elif t >= 15 and t < 15.01:
        #     return self.r_inc
        # elif t >= 20 and t < 20.01:
        #     return self.r_inc
        # elif t >= 25 and t < 25.01:
        #     return self.r_inc
        # elif t >= 30 and t < 30.01:
        #     return self.r_inc
        # elif t >= 35 and t < 35.01:
        #     return self.r_inc
        else:
            return 0
    
    # right-hand side 
    def rhs(self, y, t):      

        if t < self.T: 
            c = self.c[int(t/self.dt)]
            # print(t, self.dt)
        else: c = self.c[0]

        g, c1g, c2g, c3g, c4g = y
        dgdt = - self.r_1(c, g, c1g)
        dc1gdt = (self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g))
        dc2gdt = (self.r_2(c, c1g, c2g) - self.r_3(c, c2g, c3g))
        dc3gdt = (self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g))
        dc4gdt = self.r_4(c, c3g, c4g)
        
        return [dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt]

    def step(self):
        # initial conditions

        y0 = [self.g0, self.c1g0, self.c2g0, self.c3g0, self.c4g0]

        # integrate
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
    
        g = sol[:, 0]
        c1g = sol[:, 1]
        c2g = sol[:, 2]
        c3g = sol[:, 3]
        c4g = sol[:, 4]
        f_total = self.f_total(g, c1g, c2g, c3g, c4g)

        # print(g[-1], c1g[-1], c2g[-1], c3g[-1], c4g[-1])

        return f_total

if __name__ == "__main__":
    c = np.zeros((100000))+0.05
    encoder = FluoEncoder(c)
    fluo = encoder.step()