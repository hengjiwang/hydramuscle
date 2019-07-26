import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

class FluorescenceEncoder:
    
    # Init
    def __init__(self, c):
        self.r_inc = 200
        self.tau_ex = 1.0
        self.k1p = 2.5
        self.k1n = 0.1
        self.k2p = 16.9
        self.k2n = 205
        self.k3p = 1.1
        self.k3n = 11.8
        self.k4p = 1069
        self.k4n = 1 # 5.8
        self.c = c
        self.g0 = 3.281 # 7.4
        self.c1g0 = 4.101 # 0
        self.c2g0 = 0.017
        self.c3g0 = 0
        self.c4g0 = 0.0007
        self.phi0 = 1
        self.phi1 = 1
        self.phi2 = 1
        self.phi3 = 1
        self.phi4 = 81
        self.dt = 0.001
        self.T = len(c) * self.dt
        self.time = np.linspace(0, self.T, int(self.T/self.dt))
        self.num = len(c[0])
        
    
    # Fluorescence
    def f_total(self, g, c1g, c2g, c3g, c4g):
        f_cyt = self.phi0*g + self.phi1*c1g + self.phi2*c2g + \
        self.phi3*c3g + self.phi4*c4g
        f_cyt0 = self.phi0*self.g0 + self.phi1*self.c1g0 + \
        self.phi2*self.c2g0 + self.phi3*self.c3g0 + self.phi4*self.c4g0
        f_bg = f_cyt0 * 2.5
        return (f_cyt + f_bg) / (f_cyt0 + f_bg)
        
    # Leaky term for [Ca2+]
    def r_ex(self, c):
        return (c-self.c0)/self.tau_ex
    
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
        elif t >= 15 and t < 15.01:
            return self.r_inc
        elif t >= 20 and t < 20.01:
            return self.r_inc
        elif t >= 25 and t < 25.01:
            return self.r_inc
        elif t >= 30 and t < 30.01:
            return self.r_inc
        elif t >= 35 and t < 35.01:
            return self.r_inc
        else:
            return 0
    
    # right-hand side 
    def rhs(self, y, t):      

        if t < self.T: c = self.c[int(t/self.dt)]
        else: c = self.c[-1]

        c = np.reshape(c, self.num**2)

        g, c1g, c2g, c3g, c4g = y[            0:  self.num**2], \
                                y[  self.num**2:2*self.num**2], \
                                y[2*self.num**2:3*self.num**2], \
                                y[3*self.num**2:4*self.num**2], \
                                y[4*self.num**2:5*self.num**2]

        
        dgdt = - 1 * self.r_1(c, g, c1g)
        
        dc1gdt = 1 * (self.r_1(c, g, c1g) - self.r_2(c, c1g, c2g))
        
        dc2gdt = 1 * (self.r_2(c, c1g, c2g) - self.r_3(c, c2g, c3g))
        
        dc3gdt = 1 * (self.r_3(c, c2g, c3g) - self.r_4(c, c3g, c4g))
        
        dc4gdt = 1 * self.r_4(c, c3g, c4g)

        dydt = np.reshape([dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt], 5*self.num**2)
        
        return dydt

    def step(self):
        # initial conditions

        grid = np.ones((self.num, self.num))


        y0 = [self.g0*grid, self.c1g0*grid, self.c2g0*grid, self.c3g0*grid, self.c4g0*grid]
        y0 = np.reshape(y0, 5*self.num*self.num)

        # integrate
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
    
        g   = sol[:,             0 : self.num**2]
        c1g = sol[:,   self.num**2 : 2*self.num**2]
        c2g = sol[:, 2*self.num**2 : 3*self.num**2]
        c3g = sol[:, 3*self.num**2 : 4*self.num**2]
        c4g = sol[:, 4*self.num**2 : 5*self.num**2]
        f_total = self.f_total(g, c1g, c2g, c3g, c4g)

        return f_total
    
    def plot(self, a, b, tmin=0, tmax=-1):
        
        plt.figure(figsize=(15,4))
        plt.subplot(121)
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)])
        plt.subplot(122)
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], b[int(tmin/self.dt):int(tmax/self.dt)])
        plt.show()

if __name__ == '__main__':

    c = pd.read_csv('save/data/c_2d_diff_endo_freq.csv').values
    c = np.reshape(c, (20000, 10, 10))
    encoder = FluorescenceEncoder(c)
    fluo = encoder.step()
    df = pd.DataFrame(fluo)
    df.to_csv('save/data/fluorescence_2d_diff_endo_freq.csv', index = False)