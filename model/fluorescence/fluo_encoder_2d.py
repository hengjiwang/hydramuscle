#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from fluo_encoder import FluoEncoder

class FluoEncoder2D(FluoEncoder):
    '''An encoder that converts 2D data of [Ca2+] into fluorescence'''
    def __init__(self, c):
        # Parameters
        super().__init__(c)
        self.num = len(c[0])
    
    def rhs(self, y, t):      
        # Right-hand side formulation
        if t < self.T: c = self.c[int(t/self.dt)]
        else: c = self.c[-1]

        c = np.reshape(c, self.num**2)

        g, c1g, c2g, c3g, c4g = y[            0:  self.num**2], \
                                y[  self.num**2:2*self.num**2], \
                                y[2*self.num**2:3*self.num**2], \
                                y[3*self.num**2:4*self.num**2], \
                                y[4*self.num**2:5*self.num**2]
        
        dgdt = - self.r_1(c, g, c1g)    
        dc1gdt = self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g)    
        dc2gdt = self.r_2(c, c1g, c2g) - self.r_3(c, c2g, c3g)
        dc3gdt = self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g) 
        dc4gdt = self.r_4(c, c3g, c4g)
        dydt = np.reshape([dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt], 5*self.num**2)
        return dydt

    def step(self):
        # Time stepping
        grid = np.ones((self.num, self.num))

        y0 = [self.g0*grid, self.c1g0*grid, self.c2g0*grid, self.c3g0*grid, self.c4g0*grid]
        y0 = np.reshape(y0, 5*self.num*self.num)

        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
   
        g   = sol[:,             0 : self.num**2]
        c1g = sol[:,   self.num**2 : 2*self.num**2]
        c2g = sol[:, 2*self.num**2 : 3*self.num**2]
        c3g = sol[:, 3*self.num**2 : 4*self.num**2]
        c4g = sol[:, 4*self.num**2 : 5*self.num**2]
        f_total = self.f_total(g, c1g, c2g, c3g, c4g)

        return f_total

if __name__ == '__main__':

    c = pd.read_csv('save/data/c_2d_diff_endo_freq.csv').values
    c = np.reshape(c, (20000, 10, 10))
    encoder = FluoEncoder2D(c)
    fluo = encoder.step()
    df = pd.DataFrame(fluo)
    df.to_csv('save/data/fluorescence_2d_diff_endo_freq.csv', index = False)