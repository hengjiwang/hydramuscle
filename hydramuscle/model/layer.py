import numpy as np
from scipy.sparse import spdiags
import scipy
from collections import defaultdict

from hydramuscle.model.base_pop import BasePop
import hydramuscle.utils.utils as utils


class Layer(BasePop):
    """ A class for simulating layer-level dynamics """

    def __init__(self, cell,
                 gip3x,
                 gip3y,
                 gcx=1000,
                 gcy=1000,
                 numx=200,
                 numy=200,
                 save_interval=100,
                 active_v_beta=1,
                 stim_strength_fast=0.02):

        BasePop.__init__(self, cell, save_interval)
        self._numx = numx
        self._numy = numy
        self._num2 = numx * numy
        self._gcx = gcx
        self._gcy = gcy
        self._gip3x = gip3x
        self._gip3y = gip3y
        self.active_v_beta = active_v_beta
        self.stim_strength_fast = stim_strength_fast

        self._set_conn_mat()

    def _set_conn_mat(self):
        "Set the connetivity matrix"
        onex = np.ones(self._numx)
        Ix = np.eye(self._numx)
        oney = np.ones(self._numy)
        Iy = np.eye(self._numy)
        Dx = spdiags(np.array([onex, -2*onex, onex]),
                     np.array([-1, 0, 1]), self._numx, self._numx).toarray()
        Dy = spdiags(np.array([oney, -2*oney, oney]),
                     np.array([-1, 0, 1]), self._numy, self._numy).toarray()
        Dx[0, self._numx-1] = 1
        Dx[self._numx-1, 0] = 1
        Dy[0, 0] = -1
        Dy[self._numy-1, self._numy-1] = -1
        Dx = scipy.sparse.csr_matrix(Dx)
        Dy = scipy.sparse.csr_matrix(Dy)
        Ix = scipy.sparse.csr_matrix(Ix)
        Iy = scipy.sparse.csr_matrix(Iy)
        self._Lc = self._gcx * scipy.sparse.kron(Dx, Iy) + self._gcy * scipy.sparse.kron(Ix, Dy)
        self._Lip3 = (self._gip3x * scipy.sparse.kron(Dx, Iy) +
                      self._gip3y * scipy.sparse.kron(Ix, Dy))

    def set_stim_pattern(self, pathway, xmin, xmax, ymin, ymax, stim_times, randomnum=0, neighborsize=2):
        "Set the stimulation pattern"
        indices = utils.generate_indices(self._numy, xmin, xmax, ymin, ymax)
        # indices += random.sample(list(range(self._num2)), randomnum)
        indices += utils.generate_random_indices(self._numx, self._numy, randomnum, neighborsize=neighborsize)
        if pathway == "fast":
            self._stims_v_map[tuple(indices)] = stim_times
        elif pathway == "slow":
            self._stims_ip_map[tuple(indices)] = stim_times

    def reset_stim_pattern(self):
        self._stims_v_map = defaultdict(list)
        self._stims_ip_map = defaultdict(list)
        return

    def calc_derivs(self, y, t):
        "Calculate the derivatives based on the current-state variables"
        # Unpack dynamical variables
        num2 = self._num2
        c, s, r, ip, v, m, h, n = (y[0:num2],
                                   y[num2:2*num2],
                                   y[2*num2:3*num2],
                                   y[3*num2:4*num2],
                                   y[4*num2:5*num2],
                                   y[5*num2:6*num2],
                                   y[6*num2:7*num2],
                                   y[7*num2:8*num2])

        # Calculate terms
        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_deg = self.cell.calc_slow_terms(c, s, r, ip)
        _, i_ca, i_k, i_bk, dmdt, dhdt, dndt = self.cell.calc_fast_terms(c, v, m, h, n)

        # Update dynamical variables
        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.cell.alpha * (i_ca - self.cell._ica0)
        dsdt = self.cell.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.cell.i_plcb(self.cell.v_beta) - i_deg + self._Lip3.dot(ip)
        dvdt = - 1 / self.cell.c_m * (i_ca + i_k + i_bk) + self._Lc.dot(v)

        # Add stimulation
        for indices in self._stims_ip_map:
            dipdt[list(indices)] += (self.cell.i_plcb(self.cell.stim_slow(t, self._stims_ip_map[indices], active_v_beta=self.active_v_beta)) -
                                     self.cell.i_plcb(self.cell.v_beta))
            # print(dipdt[list(indices)])

        for indices in self._stims_v_map:
            dvdt[list(indices)] += (1 / self.cell.c_m * self.stim_strength_fast *
                                  self.cell.stim_fast(t, self._stims_v_map[indices], dur=0.01))

        # print(max(c))
        return (dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dndt)


    def _rhs(self, y, t):
        "Right-hand side"

        (dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dndt) = self.calc_derivs(y, t)

        deriv = np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dndt])
        dydt = np.reshape(deriv, len(deriv)*self._num2)

        return dydt

    def run(self, save_all=False):
        "Run the model"
        self.cell.init_fast_cell()
        self.cell.init_slow_cell()

        base_mat = np.ones((self._numy, self._numx))

        inits = [self.cell.c0,
                 self.cell.s0,
                 self.cell.r0,
                 self.cell.ip0,
                 self.cell.v0,
                 self.cell.m0,
                 self.cell.h0,
                 self.cell.n0]

        # print(inits)

        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, len(inits)*self._num2)

        # Begin counting time
        if not save_all:
            sol_ = utils.euler_odeint2(self._rhs, y0, self.T, self.dt,
                                numx=self._numx, numy=self._numy,
                                save_interval=self._save_interval,
                                layer_num=1)
        else:
            sol_ = utils.euler_odeint(self._rhs, y0, self.T, self.dt,
                                      save_interval=self._save_interval)

        return sol_

if __name__ == "__main__":
    pass
