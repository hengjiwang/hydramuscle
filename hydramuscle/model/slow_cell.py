import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from abc import abstractmethod

from hydramuscle.model.cell_base import CellBase

class SlowCell(CellBase):

    def __init__(self, T=60, dt=0.001):
        super().__init__(T, dt)
        self.k1 = 0.0004
        self.k2 = 0.08
        self.ka = 0.2
        self.kip = 0.3
        self.k3 = 0.5
        self.v40 = 0.025
        self.v41 = 0.2
        self.kr = 1
        self.k5 = 0.5
        self.k6 = 4
        self.ki = 0.2
        self.kg = 0.1 # unknown
        self.a0 = 1 # 1e-3 - 10
        self.v7 = 0.04 # 0 - 0.05
        self.v8 = None
        self.kca = 0.3
        self.k9 = 0.08
        self.beta = 20

        self.s0 = 60
        self.r0 = 0.9411764705882353
        self.ip0 = 0.01

    '''Calcium terms'''
    def i_ipr(self, c, s, ip, r):
        # Release from ER, including IP3R and leak term [uM/s]
        return (self.k2 * r * c**2 * ip**2 / (self.ka**2 + c**2) / (self.kip**2 + ip**2)) * (s - c)

    def i_serca(self, c):
        # SERCA [uM/s]
        return self.k3 * c

    def i_leak(self, c, s):
        k1 = (self.i_serca(self.c0) - self.i_ipr(self.c0, self.s0, self.ip0, self.r0)) / (self.s0 - self.c0)
        return k1 * (s - c)

    def i_pmca(self, c):
        # Additional eflux [uM/s]
        return self.k5 * c

    def i_in(self, ip):
        # Calcium entry rate [uM/s]
        return self.i_pmca(self.c0) + self.v41 * ip**2 / (self.kr**2 + ip**2) - self.v41 * self.ip0**2 / (self.kr**2 + self.ip0**2)

    '''IP3R terms'''
    def v_r(self, c, r):
        # Rates of receptor inactivation and recovery [1/s]
        return self.k6 * (self.ki**2 / (self.ki**2 + c**2) - r)

    '''IP3 terms'''
    def i_plcb(self, v8):
        # Agonist-controlled PLC-beta activity [uM/s]
        return v8 * 1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0

    def i_plcd(self, c, ip):
        # PLC-delta activity [uM/s]
        return self.v7 * c**2 / (self.kca**2 + c**2) * ip**4 / (0.05**4 + ip**4)

    def i_deg(self, ip):
        # IP3 degradion [uM/s]
        return self.k9 * ip

    '''Stimulation'''
    def stim_slow(self, t, stims, active_v8=1):
        # Stimulation

        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + 4

        return active_v8 if condition else self.v8 

    '''Numerical calculation'''
    def calc_slow_terms(self, c, s, r, ip):
        return (self.i_ipr(c, s, ip, r), 
                self.i_leak(c, s), 
                self.i_serca(c), 
                self.i_in(ip), 
                self.i_pmca(c), 
                self.v_r(c, r),
                self.i_plcd(c, ip),
                self.i_deg(ip))


    def rhs(self, y, t, stims_slow):
        # Right-hand side equations
        c, s, r, ip = y
        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.calc_slow_terms(c, s, r, ip)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca
        dsdt = self.beta*(i_serca - i_ipr -i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow)) + i_plcd - i_deg

        return [dcdt, dsdt, drdt, dipdt]

    def init_slow_cell(self):
        # Reassign some parameters to make the resting state stationary
        self.v8 = (self.i_deg(self.ip0) - self.i_plcd(self.c0, self.ip0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)
        self.in_ip0 = self.v41 * self.ip0**2 / (self.kr**2 + self.ip0**2)
        self.ipmca0 = self.i_pmca(self.c0)

    def run(self, stims=[10]):
        # Run the model
        self.init_slow_cell()

        y0 = [self.c0, self.s0, self.r0, self.ip0]
        sol = odeint(self.rhs, y0, self.time, args = (stims,), hmax = 0.005)

        return sol

if __name__ == "__main__":
    model = SlowCell(100)
    sol = model.run(stims=[10])
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]

    # Plot the results
    plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(r, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.show()

    # Plot the currents
    plt.figure()
    model.plot(model.i_ipr(c, s, ip, r), color='b')
    model.plot(model.i_serca(c), color = 'r')
    model.plot(model.i_pmca(c), color = 'g')
    model.plot(model.i_leak(c, s), color = 'y')
    plt.legend(['i_ip3r', 'i_serca', 'i_pmca', 'i_leak'])
    plt.show()


