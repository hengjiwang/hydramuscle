import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hydramuscle.model.proto_smc import ProtoSMC
from hydramuscle.model.fluo_buffer import FluoBuffer
from hydramuscle.model.euler_odeint import euler_odeint
from hydramuscle.model.force_encoder import ForceEncoder
from hydramuscle.model import plot

class SMC(ProtoSMC):
    """Smooth muscle cell with buffers"""
    def __init__(self, T=20, dt=0.001, k_ipr=0.05, s0=200, d=40e-4, v_delta=0.03, fluo_ratio=1):
        super().__init__(T, dt, k_ipr, s0, d, v_delta)
        self.fluo_buffer = FluoBuffer
        self.fluo_ratio = fluo_ratio

    def calc_fluo_terms(self, c, g, c1g, c2g, c3g, c4g):
        ir1 = self.fluo_buffer.r_1(c, g, c1g)
        ir2 = self.fluo_buffer.r_2(c, c1g, c2g)
        ir3 = self.fluo_buffer.r_3(c, c2g, c3g)
        ir4 = self.fluo_buffer.r_4(c, c3g, c4g)
        dgdt = - ir1
        dc1gdt = ir1 - ir2
        dc2gdt = ir2 - ir3
        dc3gdt = ir3 - ir4
        dc4gdt = ir4
        return ir1, ir2, ir3, ir4, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt

    def rhs(self, y, t, stims_fast, stims_slow):
        # Right-hand side formulation
        c, s, r, ip, v, m, h, n, g, c1g, c2g, c3g, c4g = y

        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.calc_slow_terms(c, s, r, ip)
        _, i_ca, i_k, i_bk, dmdt, dhdt, dndt = self.calc_fast_terms(c, v, m, h, n)
        ir1, ir2, ir3, ir4, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt = self.calc_fluo_terms(c, g, c1g, c2g, c3g, c4g)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.alpha * i_ca + self.fluo_ratio * (- ir1 - ir2 - ir3 - ir4)
        dsdt = self.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow, self.active_v_beta)) + i_plcd - i_deg
        dvdt = - 1 / self.c_m * (i_ca + i_k + i_bk - 0.002 * self.stim_fast(t, stims_fast, dur=0.01))

        return np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dndt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

    def run(self, stims_fast, stims_slow):
        # Run the model

        self.init_fast_cell()
        self.init_slow_cell()

        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.m0, self.h0, self.n0, 
        self.fluo_buffer.g0, self.fluo_buffer.c1g0, self.fluo_buffer.c2g0, self.fluo_buffer.c3g0, self.fluo_buffer.c4g0]

        sol = euler_odeint(self.rhs, y0, self.T, self.dt, stims_fast=stims_fast, stims_slow=stims_slow)

        return sol

if __name__ == '__main__':
    model = SMC(T=100, dt=0.0002, k_ipr=0.05, s0=200, d=20e-4, v_delta=0.03)
    # sol = model.run(stims_fast = [0], stims_slow = [-100])
    # sol = model.run(stims_fast = list(range(0, 20, 3))+list(range(22, 40, 4)), stims_slow = [-100])
    sol = model.run(stims_fast = [-100], stims_slow = [10])
    
    # g = sol[:,-5]
    # c1g = sol[:,-4]
    # c2g = sol[:,-3]
    # c3g = sol[:,-2]
    # c4g = sol[:,-1]
    # fluo = FluoBuffer.f_total(g, c1g, c2g, c3g, c4g)
    # force = ForceEncoder.encode(c, model.dt)

    # plot.plot_single_spike(model, sol, 0, 0.5, 0, 0.05, full_cell=True)
    plot.plot_slow_transient(model, sol, 0, 100, full_cell=True)

    
