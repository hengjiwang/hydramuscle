#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from force_encoder import KatoForceEncoder

class ForceEncoder2D(KatoForceEncoder):
    '''2D model for converting [Ca2+] into force'''
    def __init__(self, c):
        super().__init__(c)
        self.num = len(c[0])
    
    def rhs(self, y, t):
        # Right-hand side formulation
        if t < self.T: c = self.c[int(t/self.dt)]
        else: c = self.c[-1]
        
        e = self.solve_e(c)
        e = np.reshape(e, self.num**2)
        
        m1, m2, p = y[0:self.num**2], y[self.num**2:2*self.num**2], y[2*self.num**2:]
        dm1dt = - ((self.k1 + self.k3) * e + self.k2) * m1 - (self.k1 * e - self.k4) * m2 + self.k1 * e
        dm2dt = self.k3 * e * m1 - self.k4 * m2              
        dpdt = self.b * (p + self.f) * (m2/self.m2max - p) / self.lamb / (p + self.a)    
        dydt = np.reshape([dm1dt, dm2dt, dpdt], 3*self.num**2)
        
        return dydt
        
    def step(self):
        # Time stepping
        e0 = self.solve_e(self.c[0])
        m10 = 1 / (1 + self.k3 / self.k4 * e0 + 1 / (self.k1 / self.k2 * e0))
        m20 = 1 / (1 + 1 / (self.k3 / self.k4 * e0) * (1 + 1 / (self.k3 / self.k4 * e0)))
        
        e_test = self.solve_e(1e-6)

        if self.emax == None:
            self.emax = self.solve_e(np.Infinity)
            self.m2max = 1 / (1 + 1 / (self.k3 / self.k4 * self.emax) * (1 + 1 / (self.k3 / self.k4 * self.emax)))

        p0 = m20 / self.m2max
        y0 = [m10, m20, p0]
        y0 = np.reshape(y0, 3*self.num*self.num)
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        
        return sol[:,1]/self.m2max

if __name__ == '__main__':
    t = np.linspace(0, 10.0, 10000)
    c = pd.read_csv('save/data/c_2d.csv').values
    c = np.reshape(c, (20000, 10, 10))
    encoder = ForceEncoder2D(c/1e6)
    sol = encoder.step()
    force = sol[:,100:200]
    df = pd.DataFrame(force)
    df.to_csv('save/data/force_2d.csv', index = False)
