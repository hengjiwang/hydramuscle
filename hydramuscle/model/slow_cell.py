import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from abc import abstractmethod

from hydramuscle.model.cell_base import CellBase
from hydramuscle.model import plot

class SlowCell(CellBase):

    def __init__(self, T=60, dt=0.001):
        super().__init__(T, dt)
        self.k_leak = 0.0004
        self.k_ipr = 0.08
        self.ka = 0.2
        self.kip = 0.3
        self.k_serca = 0.5
        self.v_in = 0.025
        self.v_inr = 0.2
        self.kr = 1
        self.k_pmca = 0.4 # 0.5
        self.k_r = 4
        self.ki = 0.2
        self.kg = 0.1 # unknown
        self.a0 = 1 # 1e-3 - 10
        self.v_delta = 0.04 # 0 - 0.05
        self.v_beta = None
        self.kca = 0.3
        self.k_deg = 0.08
        self.beta = 20

        self.s0 = 60
        self.r0 = 0.9411764705882353
        self.ip0 = 0.01

    '''Calcium terms'''
    def i_ipr(self, c, s, ip, r):
        # Release from ER, including IP3R and leak term [uM/s]
        return (self.k_ipr * r * c**2 * ip**2 / (self.ka**2 + c**2) / (self.kip**2 + ip**2)) * (s - c)

    def i_serca(self, c):
        # SERCA [uM/s]
        return self.k_serca * c

    def i_leak(self, c, s):
        k_leak = (self.i_serca(self.c0) - self.i_ipr(self.c0, self.s0, self.ip0, self.r0)) / (self.s0 - self.c0)
        return k_leak * (s - c)

    def i_pmca(self, c):
        # Additional eflux [uM/s]
        return self.k_pmca * c

    def i_in(self, ip):
        # Calcium entry rate [uM/s]
        return self.i_pmca(self.c0) + self.v_inr * ip**2 / (self.kr**2 + ip**2) - self.v_inr * self.ip0**2 / (self.kr**2 + self.ip0**2)

    '''IP3R terms'''
    def v_r(self, c, r):
        # Rates of receptor inactivation and recovery [1/s]
        return self.k_r * (self.ki**2 / (self.ki**2 + c**2) - r)

    '''IP3 terms'''
    def i_plcb(self, v_beta):
        # Agonist-controlled PLC-beta activity [uM/s]
        return v_beta * 1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0

    def i_plcd(self, c, ip):
        # PLC-delta activity [uM/s]
        return self.v_delta * c**2 / (self.kca**2 + c**2) * ip**4 / (0.05**4 + ip**4)

    def i_deg(self, ip):
        # IP3 degradion [uM/s]
        return self.k_deg * ip

    '''Stimulation'''
    def stim_slow(self, t, stims, active_v_beta=1):
        # Stimulation

        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + 4

        return active_v_beta if condition else self.v_beta

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
        self.v_beta = (self.i_deg(self.ip0) - self.i_plcd(self.c0, self.ip0)) / (1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0)
        self.in_ip0 = self.v_inr * self.ip0**2 / (self.kr**2 + self.ip0**2)
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
    plot.plot_slow_transient(model, c, s, r, ip, 0, 100)    

