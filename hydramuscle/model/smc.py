import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hydramuscle.model.proto_smc import ProtoSMC
from hydramuscle.model.fluo_buffer import FluoBuffer
from hydramuscle.model.euler_odeint import euler_odeint
from hydramuscle.model.force_encoder import ForceEncoder

class SMC(ProtoSMC):
    """Smooth muscle cell with buffers"""
    def __init__(self, T = 20, dt = 0.001, k2 = 0.05, s0 = 200, d = 40e-4, v7 = 0.03, active_v8=1, fluo_ratio=0):
        super().__init__(T, dt, k2, s0, d, v7, active_v8)
        self.fluo_buffer = FluoBuffer
        self.fluo_ratio = fluo_ratio

        # self.v7 = 0.04
        # self.k5 = 0.05

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
        c, s, r, ip, v, m, h, bx, cx, g, c1g, c2g, c3g, c4g = y

        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.calc_slow_terms(c, s, r, ip)
        _, i_cal, i_cat, i_kca, i_bk, dmdt, dhdt, dbxdt, dcxdt = self.calc_fast_terms(c, v, m, h, bx, cx)
        ir1, ir2, ir3, ir4, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt = self.calc_fluo_terms(c, g, c1g, c2g, c3g, c4g)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.alpha * (i_cal + i_cat) + self.fluo_ratio * (- ir1 - ir2 - ir3 - ir4)
        dsdt = self.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow, self.active_v8)) + i_plcd - i_deg
        dvdt = - 1 / self.c_m * (i_cal + i_cat + i_kca + i_bk - 0.001 * self.stim_fast(t, stims_fast, dur=0.01))

        return np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

    def run(self, stims_fast, stims_slow):
        # Run the model

        self.init_fast_cell()
        self.init_slow_cell()

        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.m0, self.h0, self.bx0, self.cx0, 
        self.fluo_buffer.g0, self.fluo_buffer.c1g0, self.fluo_buffer.c2g0, self.fluo_buffer.c3g0, self.fluo_buffer.c4g0]

        sol = euler_odeint(self.rhs, y0, self.T, self.dt, stims_fast=stims_fast, stims_slow=stims_slow)

        return sol

if __name__ == '__main__':
    model = SMC(T=100, dt = 0.0002, k2 = 0.08, s0=60, v7=0., fluo_ratio=1)
    sol = model.run(stims_fast = [1,4,7,10,13,17,21,26,31,37,45], stims_slow = [-100])
    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]
    v = sol[:,4]
    g = sol[:,-5]
    c1g = sol[:,-4]
    c2g = sol[:,-3]
    c3g = sol[:,-2]
    c4g = sol[:,-1]
    fluo = FluoBuffer.f_total(g, c1g, c2g, c3g, c4g)
    force = ForceEncoder.encode(c, model.dt)

    # Plot the results
    plt.figure()
    plt.subplot(241)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(242)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(243)
    model.plot(r, ylabel = 'Inactivation ratio of IP3R')
    plt.subplot(244)
    model.plot(ip, ylabel = 'IP3[uM]')
    plt.subplot(245)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(246)
    model.plot(fluo, ylabel = 'Fluorescence')
    plt.subplot(247)
    model.plot(force, ylabel = 'Active Force')
    plt.show()

    # df = pd.DataFrame(c)
    # df.to_csv('../save/data/calcium/c_sin_ibk.csv', index = False)
