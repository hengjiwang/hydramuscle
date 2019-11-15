import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import sys, time
from tqdm import tqdm

class MHMEncoder:
    # Modified Hai-Murphy model following Maggio 2012
    def __init__(self, c, numx, numy, dt):
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
        self.dt = dt
        self.T = len(c)*self.dt
        self.time = np.linspace(0, self.T, int(self.T/self.dt))

        # Size
        self.numx = numx
        self.numy = numy
        self.num2 = numx * numy

    def rhs(self, y, t):
        # Right-hand side formulation
        if t < self.T: c = self.c[int(t/self.dt)]
        else: c = self.c[-1]

        k1 = c**self.nm / (c**self.nm + self.c_half**self.nm)
        self.k6 = k1

        m, mp, amp, am = y[0:self.num2], y[self.num2:2*self.num2], y[2*self.num2:3*self.num2], y[3*self.num2:4*self.num2]

        dmdt = -k1*m + self.k2*mp + self.k7*am
        dmpdt = k1*m + (-self.k2 - self.k3)*mp + self.k4*amp
        dampdt = self.k3*mp + (-self.k4-self.k5)*amp + self.k6*am
        damdt = self.k5*amp + (-self.k6-self.k7)*am

        dydt = np.reshape([dmdt, dmpdt, dampdt, damdt], 4*self.num2)

        return dydt

    def step(self):
        base_mat = np.ones((self.numy, self.numx))
        inits = [self.m0, self.mp0, self.amp0, self.am0]
        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, 4*self.num2) 

        # sol = odeint(self.rhs, y0, self.time, hmax = 0.005)

        # Begin counting time
        start_time = time.time() 
        
        y = y0
        T = self.T
        dt = self.dt

        sol = np.zeros((int(T/dt), 4*self.num2))

        # Euler method integration
        for j in tqdm(np.arange(0, int(T/dt))):
            t = j*dt
            dydt = self.rhs(y, t)
            y += dydt * dt
            sol[j, :] = y
        
        # End counting time
        elapsed = (time.time() - start_time) 


        return self.K * (sol[:,2*self.num2:3*self.num2] + sol[:,3*self.num2:4*self.num2])


if __name__ == "__main__":
    c = pd.read_csv("../../save/data/calcium/c_200x200_100s_cb.csv").values
    encoder = MHMEncoder(c, 200, 200, 0.02)
    force = encoder.step()
    df = pd.DataFrame(force)
    df.to_csv('force_200x200_100s_cb.csv', index = False)
