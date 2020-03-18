import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from hydramuscle.model.cell_base import CellBase
from hydramuscle.model.euler_odeint import euler_odeint

class FastCell(CellBase):
    
    def __init__(self, T = 20, dt = 0.001):

        super().__init__(T,dt)

        # General parameters
        self.c_m = 1e-6 # [F/cm^2]
        self.A_cyt = 4e-5 # [cm^2]
        self.V_cyt = 6e-9 # [cm^3]
        self.d = 20e-4 # [cm]
        self.F = 96485332.9 # [mA*s/mol]

        self.alpha = 1e9 / (2 * self.F * self.d)

        self.v0 = -50 # (-40 to -60)
        self.m0 = 0
        self.h0 = 0
        self.n0 = 0

        # Calcium leak
        self.tau_ex = 0.1 # [s]
        
        # Ca channel parameters
        self.g_ca = 0.0005 # [S/cm^2] 
        self.e_ca = 51

        # BK parameters
        self.g_k = 0.0025
        self.e_k = -75 

        # Background parameters
        self.g_bk = 0
        self.e_bk = -55


    def sig(self, v, vstar, sstar):
        "Sigmoidal function"
        return 1 / (1 + np.exp(-(v-vstar)/sstar))

    def bell(self, v, vstar, sstar, taustar, tau0):
        "Bell-shape function"
        return taustar/(np.exp(-(v-vstar)/sstar) + np.exp((v-vstar)/sstar)) + tau0
    
    ### Ca channel terms (Diderichsen 2006)
    def i_ca(self, v, m, h):
        "Current through calcium channel [mA/cm^2]"
        return self.g_ca * m**2 * h * (v - self.e_ca)

    def m_inf(self, v):
        return self.sig(v, -25, 10)

    def h_inf(self, v):
        return self.sig(v, -28, -5)

    def tau_m(self, v):
        return self.bell(v, -23, 20, 0.001, 0.00005)

    def tau_h(self, v):
        return self.bell(v, 0, 20, 0.03, 0.021)


    ### K channel terms (Diderichsen 2006)
    def i_k(self, v, n):
        "Current through potassium channel [mA/cm^2]"
        return self.g_k * n**4 * (v - self.e_k)

    def n_inf(self, v):
        return self.sig(v, -18.5, 23)

    def tau_n(self, v):
        return self.bell(v, -10, 25, 0.0015, 0.015)

    ### Leaky current
    def i_bk(self, v):
        "Background voltage leak [mA/cm^2]"
        return self.g_bk * (v - self.e_bk)


    def r_ex(self, c):
        "Calcium terms"
        return (c-self.c0)/self.tau_ex

    
    def stim_fast(self, t, stims, dur=0.01):
        "Stimulation"
        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + dur

       	return int(condition)

    ### Numerical terms
    def calc_fast_terms(self, c, v, m, h, n):
        return (self.r_ex(c), 
                self.i_ca(v, m, h), 
                self.i_k(v, n),
                self.i_bk(v),
                (self.m_inf(v) - m)/self.tau_m(v),
                (self.h_inf(v) - h)/self.tau_h(v),
                (self.n_inf(v) - n)/self.tau_n(v))

    def rhs(self, y, t, stims_fast):
        "Right-hand side equations"
        c, v, m, h, n = y

        r_ex, i_ca, i_k, i_bk, dmdt, dhdt, dndt = self.calc_fast_terms(c, v, m, h, n)

        dcdt = -r_ex + self.alpha*(-i_ca + self.ica0)
        dvdt = - 1 / self.c_m * (i_ca+ i_k + i_bk - 0.002 * self.stim_fast(t, stims_fast, dur=0.005))

        return np.array([dcdt, dvdt, dmdt, dhdt, dndt])


    def init_fast_cell(self):
        "Reassign some parameters to make the resting state stationary"
        self.m0 = self.m_inf(self.v0)
        self.h0 = self.h_inf(self.v0)
        self.n0 = self.n_inf(self.v0)
        self.ica0 = self.i_ca(self.v0, self.m0, self.h0)
        self.ik0 = self.i_k(self.v0, self.n0)
        self.g_bk = - (self.ica0 + self.ik0)/(self.v0 - self.e_bk)

    def run(self, stims_fast):
        "Run the model"

        self.init_fast_cell()

        y0 = [self.c0, self.v0, self.m0, self.h0, self.n0]

        sol = euler_odeint(self.rhs, y0, self.T, self.dt, stims_fast=stims_fast)

        return sol

if __name__ == '__main__':
    model = FastCell(20, 0.0002)
    sol = model.run([1,3,5,7,9,11,13,15])
    # sol = model.run([2])
    c = sol[:, 0]
    v = sol[:, 1]
    m = sol[:, 2]
    h = sol[:, 3]
    n = sol[:, 4]

    # Plot the results
    tmin = 1
    tmax = 1.5
    index_min = int(tmin/model.dt)
    index_max = int(tmax/model.dt)

    plt.figure(figsize=(30,10), tight_layout=True)

    ax1 = plt.subplot2grid((1,3), (0,0), colspan=1)
    ax1.plot(model.time[index_min:index_max]*1000, c[index_min:index_max], linewidth=5, color="k")
    ax1.tick_params(labelsize=20)
    ax1.set_xlabel("time(ms)", fontsize=20)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=20)
    ax1.text(-0.01, 1.05, 'A', size=40, weight="bold", transform=ax1.transAxes)

    ax2 = plt.subplot2grid((1,3), (0,1), colspan=1)
    ax2.plot(model.time[index_min:index_max]*1000, v[index_min:index_max], linewidth=5, color="k")
    ax2.tick_params(labelsize=20)
    ax2.set_xlabel("time(ms)", fontsize=20)
    ax2.set_ylabel("Membrane voltage(mV)", fontsize=20)
    ax2.text(-0.01, 1.05, 'B', size=40, weight="bold", transform=ax2.transAxes)

    tmin = 1
    tmax = 1.05
    index_min = int(tmin/model.dt)
    index_max = int(tmax/model.dt)

    ax3 = plt.subplot2grid((1,3), (0,2), colspan=1)
    ax3.plot(model.time[index_min:index_max]*1000, model.i_ca(v, m, h)[index_min:index_max], linewidth=5, color="r", label=r"I$_{Ca}$")
    ax3.plot(model.time[index_min:index_max]*1000, model.i_k(v, n)[index_min:index_max], linewidth=5, color="b", label=r"I$_{K}$")
    ax3.plot(model.time[index_min:index_max]*1000, model.i_bk(v)[index_min:index_max], linewidth=5, color="purple", linestyle="--", label=r"I$_{b}$")
    ax3.legend(fontsize=30)
    ax3.tick_params(labelsize=20)
    ax3.set_xlabel("time(ms)", fontsize=20)
    ax3.set_ylabel(r"Membrane current(mA/cm$^2$)", fontsize=20)
    ax3.text(-0.005, 1.05, 'C', size=40, weight="bold", transform=ax3.transAxes)
    ax3.bar(1000, 0.0005, width=5, bottom=-0.015, align='edge', color='k')

    plt.show()
    