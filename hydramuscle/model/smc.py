import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hydramuscle.model.fast_cell import FastCell
from hydramuscle.model.slow_cell import SlowCell
from hydramuscle.model.force_encoder_ecto import ForceEncoderEcto
from hydramuscle.model.force_encoder_endo import ForceEncoderEndo
from hydramuscle.model.euler_odeint import euler_odeint
from hydramuscle.model import plot

class SMC(SlowCell, FastCell):

    def __init__(self, T = 20, dt = 0.001, k_ipr = 0.05, s0 = 60, d = 20e-4, v_delta = 0.04, active_v_beta=1):
        SlowCell.__init__(self, T, dt)
        FastCell.__init__(self, T, dt)

        self.k_ipr = k_ipr
        self.s0 = s0
        self.d = d
        self.v_delta = v_delta
        self.alpha = 1e9 / (2 * self.F * self.d)
        self.active_v_beta = active_v_beta


    def i_in(self, ip):
        return self.alpha * (self.ica0) + self.ipmca0 + self.v_inr * ip**2 / (self.kr**2 + ip**2) - self.in_ip0


    def rhs(self, y, t, stims_fast, stims_slow):
        # Right-hand side formulation
        c, s, r, ip, v, m, h, n = y

        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.calc_slow_terms(c, s, r, ip)
        _, i_ca, i_k, i_bk, dmdt, dhdt, dndt = self.calc_fast_terms(c, v, m, h, n)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.alpha * i_ca
        dsdt = self.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow, self.active_v_beta)) + i_plcd - i_deg

        dvdt = - 1 / self.c_m * (i_ca + i_k + i_bk - 0.001 * self.stim_fast(t, stims_fast, dur=0.01))

        return np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dndt])


    def run(self, stims_fast, stims_slow, T=None, dt=None):
        # Run the model
        self.init_fast_cell()
        self.init_slow_cell()

        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.m0, self.h0, self.n0]

        # y = y0
        # T = self.T
        # dt = self.dt]
        
        if not T:   T = self.T
        if not dt:  dt = self.dt

        sol = euler_odeint(self.rhs, y0, T, dt, stims_fast=stims_fast, stims_slow=stims_slow)

        return sol

if __name__ == '__main__':
    model = SMC(T=200, dt=0.0002, k_ipr=0.02, s0=100, d=20e-4, v_delta=0.03)

    ### One Fast Spike ###
    # sol = model.run(stims_fast = [0], stims_slow = [-100])
    # plot.plot_single_spike(model, sol, 0, 2, 0, 0.05, full_cell=True)


    ### One Slow Transient ###
    # sol = model.run(stims_fast = [-100], stims_slow = [10])
    # plot.plot_slow_transient(model, sol, 0, 100, full_cell=True)

    ### Multiple Fast Spikes ###
    sol = model.run(stims_fast=[0, 5.2, 8.2, 10.6, 12.8, 15, 17.3, 19.4, 21.9, 25.1, 29.5, 34.3,
                                100, 105.7, 108.8, 111.6, 113.8, 116.1, 118.3, 121, 124.2, 129, 135.4], 
                    stims_slow=[-100])
    force_ecto = ForceEncoderEcto.encode(sol[:, 0], model.dt);
    force_endo = ForceEncoderEndo.encode(sol[:, 0], model.dt);
    plot.plot_multiple_spikes(model, sol, force_ecto, force_endo, 0, 100, 0, 500)
    # df_ecto = pd.DataFrame(force_ecto)
    # df_endo = pd.DataFrame(force_endo)