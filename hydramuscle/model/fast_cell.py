import sys, os
sys.path.append('../..')

import numpy as np

from hydramuscle.model.base_cell import BaseCell
import hydramuscle.utils.utils as utils
import hydramuscle.utils.plot as plot


class FastCell(BaseCell):

    def __init__(self, T=20, dt=0.001):
        super().__init__(T, dt)

        # General parameters
        self.c_m = 1e-6  # [F/cm^2]
        self._A_cyt = 4e-5  # [cm^2]
        self._V_cyt = 6e-9  # [cm^3]
        self._d = 20e-4  # [cm]
        self._F = 96485332.9  # [mA*s/mol]

        self.alpha = 1e9 / (2 * self._F * self._d)

        self.v0 = -50  # (-40 to -60)
        self.m0 = 0
        self.h0 = 0
        self.n0 = 0
        self._ica0 = 0
        self._ik0 = 0

        # Calcium leak
        self._tau_ex = 0.1  # [s]

        # Ca channel parameters
        self._g_ca = 0.0005  # [S/cm^2]
        self._e_ca = 51

        # K channel parameters
        self._g_k = 0.0025
        self._e_k = -75

        # Background parameters
        self._g_bk = 0
        self._e_bk = -55

    # Ca channel terms (Diderichsen 2006)
    def i_ca(self, v, m, h):
        "Current through calcium channel [mA/cm^2]"
        return self._g_ca * m ** 2 * h * (v - self._e_ca)

    def _m_inf(self, v):
        return utils.sig(v, -25, 10)

    def _h_inf(self, v):
        return utils.sig(v, -28, -5)

    def _tau_m(self, v):
        return utils.bell(v, -23, 20, 0.001, 0.00005)

    def _tau_h(self, v):
        return utils.bell(v, 0, 20, 0.03, 0.021)

    # K channel terms (Diderichsen 2006)
    def i_k(self, v, n):
        "Current through potassium channel [mA/cm^2]"
        return self._g_k * n ** 4 * (v - self._e_k)

    def _n_inf(self, v):
        return utils.sig(v, -18.5, 23)

    def _tau_n(self, v):
        return utils.bell(v, -10, 25, 0.0015, 0.015)

    # Leaky current
    def i_bk(self, v):
        "Background voltage leak [mA/cm^2]"
        return self._g_bk * (v - self._e_bk)

    def _r_ex(self, c):
        "Calcium terms"
        return (c - self.c0) / self._tau_ex

    def stim_fast(self, t, stims, dur=0.01):
        "Stimulation"
        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + dur

        return int(condition)

    # Numerical terms
    def calc_fast_terms(self, c, v, m, h, n):
        return (self._r_ex(c),
                self.i_ca(v, m, h),
                self.i_k(v, n),
                self.i_bk(v),
                (self._m_inf(v) - m) / self._tau_m(v),
                (self._h_inf(v) - h) / self._tau_h(v),
                (self._n_inf(v) - n) / self._tau_n(v))

    def _rhs(self, y, t, stims_fast):
        "Right-hand side equations"
        c, v, m, h, n = y

        r_ex, i_ca, i_k, i_bk, dmdt, dhdt, dndt = self.calc_fast_terms(c, v, m, h, n)

        dcdt = -r_ex + self.alpha * (-i_ca + self._ica0)
        dvdt = - 1 / self.c_m * (i_ca + i_k + i_bk - 0.002 * self.stim_fast(t, stims_fast, dur=0.005))

        return np.array([dcdt, dvdt, dmdt, dhdt, dndt])

    def init_fast_cell(self):
        "Reassign some parameters to make the resting state stationary"
        self.m0 = self._m_inf(self.v0)
        self.h0 = self._h_inf(self.v0)
        self.n0 = self._n_inf(self.v0)
        self._ica0 = self.i_ca(self.v0, self.m0, self.h0)
        self._ik0 = self.i_k(self.v0, self.n0)
        self._g_bk = - (self._ica0 + self._ik0) / (self.v0 - self._e_bk)

    def run(self, stims_fast):
        "Run the model"

        self.init_fast_cell()
        y0 = [self.c0, self.v0, self.m0, self.h0, self.n0]
        sol_ = utils.euler_odeint(self._rhs, y0, self.T, self.dt, stims_fast=stims_fast)

        return sol_


if __name__ == '__main__':
    model = FastCell(0.2, 0.0002)
    # sol = model.run([1,3,5,7,9,11,13,15])
    sol = model.run([0, 0.1])

    # Plot the results
    plot.plot_single_spike(model, sol, 0, 100, 0, 0.05)

