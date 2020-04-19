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
        self._k_leak = 0.0004
        self._k_ipr = 0.08
        self._ka = 0.2
        self._kip = 0.3
        self._k_serca = 0.5
        self._v_in = 0.025
        self._v_inr = 0.2
        self._kr = 1
        self._k_pmca = 0.4 # 0.5
        self._k_r = 4
        self._ki = 0.2
        self._kg = 0.1 # unknown
        self._a0 = 1 # 1e-3 - 10
        self._v_delta = 0.04 # 0 - 0.05
        self.v_beta = None
        self._kca = 0.3
        self._k_deg = 0.08
        self.beta = 20

        self.s0 = 60
        self.r0 = 0.9411764705882353
        self.ip0 = 0.01

        self._in_ip0 = None
        self._ipmca0 = None

    # Calcium terms
    def i_ipr(self, c, s, ip, r):
        "Release from ER, including IP3R and leak term [uM/s]"
        return (self._k_ipr * r * c**2 * ip**2 / (self._ka**2 + c**2) / (self._kip**2 + ip**2)) * (s - c)

    def i_serca(self, c):
        "SERCA [uM/s]"
        return self._k_serca * c

    def i_leak(self, c, s):
        k_leak = (self.i_serca(self.c0) - self.i_ipr(self.c0, self.s0, self.ip0, self.r0)) / (self.s0 - self.c0)
        return k_leak * (s - c)

    def i_pmca(self, c):
        "Additional eflux [uM/s]"
        return self._k_pmca * c

    def i_in(self, ip):
        "Calcium entry rate [uM/s]"
        return self.i_pmca(self.c0) + self._v_inr * ip**2 / (self._kr**2 + ip**2) - self._v_inr * self.ip0**2 / (self._kr**2 + self.ip0**2)

    # IP3R terms
    def v_r(self, c, r):
        "Rates of receptor inactivation and recovery [1/s]"
        return self._k_r * (self._ki**2 / (self._ki**2 + c**2) - r)

    # IP3 terms
    def i_plcb(self, v_beta):
        "Agonist-controlled PLC-beta activity [uM/s]"
        return v_beta * 1 / ((1 + self._kg)*(self._kg/(1+self._kg) + self._a0)) * self._a0

    def i_plcd(self, c, ip):
        "PLC-delta activity [uM/s]"
        return self._v_delta * c**2 / (self._kca**2 + c**2) * ip**4 / (0.05**4 + ip**4)

    def i_deg(self, ip):
        "IP3 degradion [uM/s]"
        return self._k_deg * ip

    # Stimulation
    def stim_slow(self, t, stims, active_v_beta=1):
        "Stimulation"

        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + 4

        return active_v_beta if condition else self.v_beta

    # Numerical calculation
    def calc_slow_terms(self, c, s, r, ip):
        return (self.i_ipr(c, s, ip, r), 
                self.i_leak(c, s), 
                self.i_serca(c), 
                self.i_in(ip), 
                self.i_pmca(c), 
                self.v_r(c, r),
                self.i_plcd(c, ip),
                self.i_deg(ip))


    def _rhs(self, y, t, stims_slow):
        "Right-hand side equations"
        c, s, r, ip = y
        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.calc_slow_terms(c, s, r, ip)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca
        dsdt = self.beta*(i_serca - i_ipr -i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow)) + i_plcd - i_deg

        return [dcdt, dsdt, drdt, dipdt]

    def init_slow_cell(self):
        "Reassign some parameters to make the resting state stationary"
        self.v_beta = ((self.i_deg(self.ip0) -
                        self.i_plcd(self.c0, self.ip0)) /
                        (1 / ((1 + self._kg) *
                            (self._kg/(1 + self._kg) +
                            self._a0)) *
                        self._a0))
        self._in_ip0 = self._v_inr * self.ip0**2 / (self._kr**2 + self.ip0**2)
        self._ipmca0 = self.i_pmca(self.c0)

    def run(self, stims_slow=[10]):
        "Run the model"
        self.init_slow_cell()
        y0 = [self.c0, self.s0, self.r0, self.ip0]
        sol_ = odeint(self._rhs, y0, self.time, args=(stims_slow,), hmax=0.005)

        return sol_

if __name__ == "__main__":
    model = SlowCell(100)
    sol = model.run(stims_slow=[10])

    # Plot the results
    plot.plot_slow_transient(model, sol, 0, 100)

