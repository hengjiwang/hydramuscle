import numpy as np

from hydramuscle.model.fast_cell import FastCell
from hydramuscle.model.slow_cell import SlowCell
import hydramuscle.utils.plot as plot
import hydramuscle.utils.utils as utils


class SMC(SlowCell, FastCell):

    def __init__(self, **kwargs):
        SlowCell.__init__(self, kwargs['T'], kwargs['dt'])
        FastCell.__init__(self, kwargs['T'], kwargs['dt'])

        for attr in kwargs:
            utils.set_attr(self, attr, kwargs[attr])

        self.alpha = 1e9 / (2 * self._F * self._d)
        self._active_v_beta = 1

    def i_in(self, ip):
        return self.alpha * (self._ica0) + self._ipmca0 + self._v_inr * ip ** 2 / (
                    self._kr ** 2 + ip ** 2) - self._in_ip0

    def _rhs(self, y, t, stims_fast, stims_slow):
        "Right-hand side formulation"
        c, s, r, ip, v, m, h, n = y

        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_deg = self.calc_slow_terms(c, s, r, ip)
        _, i_ca, i_k, i_bk, dmdt, dhdt, dndt = self.calc_fast_terms(c, v, m, h, n)

        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.alpha * i_ca
        dsdt = self.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.i_plcb(self.stim_slow(t, stims_slow, self._active_v_beta)) - i_deg

        dvdt = - 1 / self.c_m * (i_ca + i_k + i_bk - 0.001 * self.stim_fast(t, stims_fast, dur=0.01))

        return np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dndt])

    def run(self, stims_fast, stims_slow, T=None, dt=None):
        "Run the model"
        self.init_fast_cell()
        self.init_slow_cell()

        y0 = [self.c0, self.s0, self.r0, self.ip0, self.v0, self.m0, self.h0, self.n0]

        if not T:
            T = self.T
        if not dt:
            dt = self.dt

        sol_ = utils.euler_odeint(self._rhs, y0, T, dt, stims_fast=stims_fast, stims_slow=stims_slow)

        return sol_


if __name__ == '__main__':
    model = SMC(T=100, dt=0.0002, k_ipr=0.08, s0=60, d=10e-4, k_deg=0.4)

    ### One Fast Spike ###
    # sol = model.run(stims_fast = [0], stims_slow = [-100])
    # plot.plot_single_spike(model, sol, 0, 100, 0, 0.05, full_cell=True)

    sol = model.run(stims_fast=[0, 5.2, 8.2, 10.6, 12.8, 15, 17.3, 19.4, 21.9, 25.1, 29.5, 34.3], stims_slow=[-100])
    plot.plot_single_spike(model, sol, 0, 200, 0, 0.05, full_cell=True)

    ### One Slow Transient ###
    # sol = model.run(stims_fast = [-100], stims_slow = [10])
    # plot.plot_slow_transient(model, sol, 0, 100, full_cell=True)

    ### Multiple Fast Spikes ###
    # sol = model.run(stims_fast=[0, 5.2, 8.2, 10.6, 12.8, 15, 17.3, 19.4, 21.9, 25.1, 29.5, 34.3,
    #                             100, 105.7, 108.8, 111.6, 113.8, 116.1, 118.3, 121, 124.2, 129, 135.4],
    #                 stims_slow=[-100])
    # force_ecto = ForceEncoderEcto.encode(sol[:, 0], model.dt);
    # force_endo = ForceEncoderEndo.encode(sol[:, 0], model.dt);
    # plot.plot_multiple_spikes(model, sol, force_ecto, force_endo, 0, 100, 0, 500)