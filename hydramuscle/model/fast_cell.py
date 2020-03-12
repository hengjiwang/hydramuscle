import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from hydramuscle.model.cell_base import CellBase
from hydramuscle.model.euler_odeint import euler_odeint

class FastCell(CellBase):

    def __init__(self, T = 20, dt = 0.001, gkca=10e-9):

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
        self.bx0 = 0
        self.cx0 = 0

        # Calcium leak
        self.tau_ex = 0.1 # [s]
        
        # CaL parameters
        self.g_cal = 0.0005 # [S/cm^2] 
        self.e_cal = 51
        self.k_cal = 1 # [uM]

        # CaT parameters
        self.g_cat = 0.003 # 0.0003
        self.e_cat = 51

        # BK parameters
        self.gkca = gkca
        self.g_kca = self.gkca / self.A_cyt # 10e-9 / self.A_cyt # 45.7e-9 / self.A_cyt
        self.e_k = -75 

        # Background parameters
        self.g_bk = 0
        self.e_bk = -53 # -55


    def sig(self, v, vstar, sstar):
        "Sigmoidal function"
        return 1 / (1 + np.exp(-(v-vstar)/sstar))

    def bell(self, v, vstar, sstar, taustar, tau0):
        "Bell-shape function"
        return taustar/(np.exp(-(v-vstar)/sstar) + np.exp((v-vstar)/sstar)) + tau0
    
    ### CaL channel terms (Diderichsen 2006)
    def i_cal(self, v, m, h):
        "L-type calcium channel [mA/cm^2]"
        return self.g_cal * m**2 * h * (v - self.e_cal)

    def m_inf(self, v):
        return self.sig(v, -25, 10)

    def h_inf(self, v):
        return self.sig(v, -28, -5)

    def tau_m(self, v):
        return self.bell(v, -23, 20, 0.001, 0.00005)

    def tau_h(self, v):
        return self.bell(v, 0, 20, 0.03, 0.021)

    ### T-type calcium channel terms (Mahapatra 2018)
    def i_cat(self, v, bx, cx):
        "T-type calcium channel [mA/cm^2]"
        return self.g_cat * bx**2 * cx * (v - self.e_cat)

    def bx_inf(self, v):
        return self.sig(v, -32.1, 6.9)

    def cx_inf(self, v):
        return self.sig(v, -63.8, -5.3)

    def tau_bx(self, v):
        return 0.00045 + 0.0039 / (1 + ((v+66)/26)**2)

    def tau_cx(self, v):
        return 0.15 - 0.15 / ((1 + np.exp((v-417.43)/203.18))*(1 + np.exp(-(v+61.11)/8.07)))
    
    def i_kca(self, v, c):
        "BK channel (Corrias 2007)"
        if isinstance(c, float) and c<=0:
            raise ValueError('[Ca2+] should be larger than 0')
        return self.g_kca * 1 / (1 + np.exp(v/(-17) - 2 * np.log(c))) * (v - self.e_k)
        # return self.g_kca * c**2 / (c**2 + 0.8**2) * (v - self.e_k)


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
    def calc_fast_terms(self, c, v, m, h, bx, cx):
        return (self.r_ex(c), 
                self.i_cal(v, m, h), 
                self.i_cat(v, bx, cx),
                self.i_kca(v, c),
                self.i_bk(v),
                (self.m_inf(v) - m)/self.tau_m(v),
                (self.h_inf(v) - h)/self.tau_h(v),
                (self.bx_inf(v) - bx)/self.tau_bx(v),
                (self.cx_inf(v) - cx)/self.tau_cx(v))

    def rhs(self, y, t, stims_fast):
        "Right-hand side equations"
        c, v, m, h, bx, cx = y

        r_ex, i_cal, i_cat, i_kca, i_bk, dmdt, dhdt, dbxdt, dcxdt = self.calc_fast_terms(c, v, m, h, bx, cx)

        dcdt = -r_ex + self.alpha*(-i_cal + self.ical0 - i_cat + self.icat0)
        dvdt = - 1 / self.c_m * (i_cal + i_cat + i_kca + i_bk - 0.002 * self.stim_fast(t, stims_fast, dur=0.01))

        return np.array([dcdt, dvdt, dmdt, dhdt, dbxdt, dcxdt])


    def init_fast_cell(self):
        "Reassign some parameters to make the resting state stationary"
        self.m0 = self.m_inf(self.v0)
        self.h0 = self.h_inf(self.v0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)
        self.ical0 = self.i_cal(self.v0, self.m0, self.h0)
        self.icat0 = self.i_cat(self.v0, self.bx0, self.cx0)
        self.ikca0 = self.i_kca(self.v0, self.c0)
        self.g_bk = - (self.ical0 + self.icat0 + self.ikca0)/(self.v0 - self.e_bk)

    def run(self, stims_fast):
        "Run the model"

        self.init_fast_cell()

        y0 = [self.c0, self.v0, self.m0, self.h0, self.bx0, self.cx0]

        sol = euler_odeint(self.rhs, y0, self.T, self.dt, stims_fast=stims_fast)

        return sol

if __name__ == '__main__':
    model = FastCell(20, 0.0002)
    sol = model.run([1,3,5,7,9,11,13,15])
    c = sol[:, 0]
    v = sol[:, 1]
    m = sol[:, 2]
    h = sol[:, 3]
    bx = sol[:, 4]
    cx = sol[:, 5]

    # Plot the results
    plt.figure()
    plt.subplot(421)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(422)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(423)
    model.plot(model.i_cal(v, m, h), ylabel='i_cal[mA/cm^2]')
    plt.subplot(424)
    model.plot(model.i_cat(v, bx, cx), ylabel='i_cat[mA/cm^2]')
    plt.subplot(425)
    model.plot(model.i_bk(v), ylabel='i_bk[mA/cm^2]')
    plt.subplot(426)
    model.plot(model.i_kca(v, c), ylabel = 'i_kca[mA/cm^2]')
    plt.subplot(427)
    model.plot([0.002 * model.stim_fast(t, [1,3,5,7,9,11,13,15], dur=0.01) for t in model.time], ylabel='i_stim[mA/cm^2]', color = 'r')
    plt.show()
    

    
