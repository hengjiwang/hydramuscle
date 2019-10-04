#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import sys

class MHMEncoder:
    # Modified Hai-Murphy model following Maggio 2012
    def __init__(self, c):
        self.c = c

        # Attachment & Detachment rates
        self.k2 = 0.1399
        self.k3 = 14.4496
        self.k4 = 3.6124
        self.k5 = self.k2
        self.k6 = 0
        self.k7 = 0.1340

        # General parameters
        self.nm = 4.7135
        self.c_half = 0.4640758 # 1
        self.K = 5.0859 # 1 

        # Initial variables
        self.m0 = 1
        self.mp0 = 0
        self.amp0 = 0
        self.am0 = 0

        # Time parameters
        self.dt = 0.001
        self.T = len(c)*self.dt
        self.time = np.linspace(0, self.T, int(self.T/self.dt))

    def rhs(self, y, t):
        # Right-hand side formulation
        if t < self.T: c = self.c[int(t/self.dt)]
        else: c = self.c[-1]

        k1 = c**self.nm / (c**self.nm + self.c_half**self.nm)
        self.k6 = k1

        trans = np.array([[-k1, self.k2, 0, self.k7], 
                          [k1, -self.k2 - self.k3, self.k4, 0],
                          [0, self.k3, -self.k4-self.k5, self.k6],
                          [0, 0, self.k5, - self.k6 - self.k7]])

        return list(trans@np.array(y))

    def step(self):
        y0 = [self.m0, self.mp0, self.amp0, self.am0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return self.K * (sol[:,2] + sol[:,3])
