#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class FastCell:
    '''An intracellular model for calcium influx from extracellular space'''
    def __init__(self, T = 20, dt = 0.001):
        # General parameters
        self.c_m = 1e-6 # [F/cm^2]
        self.A_cyt = 4e-5 # [cm^2]
        self.V_cyt = 6e-9 # [cm^3]
        self.d = 10e-4 # 10e-4 # [cm]
        self.F = 96485332.9 # [mA*s/mol]
        self.c0 = 0.05
        self.v0 = -50 # (-40 to -60)
        self.n0 = None
        self.hv0 = None
        self.hc0 = None
        self.x0 = None
        self.z0 = None
        self.p0 = None
        self.q0 = None
        self.bx0 = None
        self.cx0 = None

        # Calcium leak
        self.tau_ex = 0.1 # [s]
        
        # CaL parameters
        self.g_cal = 0.0005 # [S/cm^2] 
        self.e_cal = 51
        self.k_cal = 1 # [uM]

        # CaT parameters
        self.g_cat = 0.003 # 0.0003
        self.e_cat = 51

        # KCNQ parameters
        self.g_kcnq = 0 # 0.0001 # [S/cm^2]
        self.e_k = -75 

        # Kv parameters
        self.g_kv = 0 # 0.0004

        # BK parameters
        self.g_kca =  10e-9 / self.A_cyt # 45.7e-9 / self.A_cyt

        # Background parameters
        self.g_bk = None
        self.e_bk = -55

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''CaL channel terms (Mahapatra 2018)'''
    def i_cal(self, v, n, hv, hc):
        # L-type calcium channel [mA/cm^2]
        return self.g_cal * n**2 * hv * hc * (v - self.e_cal)

    def n_inf(self, v):
        # [-]
        return 1 / (1 + np.exp(-(v+9)/8))

    def hv_inf(self, v):
        # [-]
        return 1 / (1 + np.exp((v+30)/13))

    def hc_inf(self, c):
        # [-]
        return self.k_cal / (self.k_cal + c)

    def tau_n(self, v):
        # [s]
        return 0.000001 / (1 + np.exp(-(v+22)/308))

    def tau_hv(self, v):
        # [s]
        return 0.09 * (1 - 1 / ((1 + np.exp((v+14)/45)) * (1 + np.exp(-(v+9.8)/3.39))))

    def tau_hc(self):
        # [s]
        return 0.02

    '''T-type calcium channel terms (Mahapatra 2018)'''
    def i_cat(self, v, bx, cx):
        return self.g_cat * bx**2 * cx * (v - self.e_cat)

    def bx_inf(self, v):
        return 1 / (1 + np.exp(-(v+32.1)/6.9))

    def cx_inf(self, v):
        return 1 / (1 + np.exp((v+63.8)/5.3))

    def tau_bx(self, v):
        return 0.00045 + 0.0039 / (1 + ((v+66)/26)**2)

    def tau_cx(self, v):
        return 0.15 - 0.15 / ((1 + np.exp((v-417.43)/203.18))*(1 + np.exp(-(v+61.11)/8.07)))
    
    '''KCNQ1 channel terms (Mahapatra 2018)'''
    def i_kcnq(self, v, x, z):
        # KCNQ1 channel [mA/cm^2]
        return self.g_kcnq * x**2 * z * (v - self.e_k)

    def x_inf(self, v):
        # [-]
        return 1 / (1 + np.exp(-(v+7.1)/15))

    def z_inf(self, v):
        # [-]
        return 0.55 / (1 + np.exp((v+55)/9)) + 0.45

    def tau_x(self, v):
        # [s]
        return 0.001 / (1 + np.exp((v+15)/20))

    def tau_z(self, v):
        # [s]
        return 0.01 * (200 + 10 / (1 + 2 * ((v+56)/120)**5))

    '''Kv channels (Mahapatra 2018)'''
    def i_kv(self, v, p, q):
        return self.g_kv * p * q * (v - self.e_k)
    
    def p_inf(self, v):
        return 1 / (1 + np.exp(-(v+1.1)/11))

    def q_inf(self, v):
        return 1 / (1 + np.exp((v+58)/15))

    def tau_p(self, v):
        return 0.001 / (1 + np.exp((v+15)/20))

    def tau_q(self, v):
        return 0.4 * (200 + 10 / (1 + 2*((v+54.18)/120)**5))

    '''BK channel terms (Corrias 2007)'''
    def i_kca(self, v, c):
        return self.g_kca * 1 / (1 + np.exp(v/(-17) - 2 * np.log(c))) * (v - self.e_k)
        # return 5 * self.g_kca * c**2 / (c**2 + 5**2) * (v - self.e_k)

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        g_bk = - (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) \
        + self.i_cat(self.v0, self.bx0, self.cx0) \
        + self.i_kcnq(self.v0, self.x0, self.z0) \
        + self.i_kv(self.v0, self.p0, self.q0) \
        + self.i_kca(self.v0, self.c0))/(self.v0 - self.e_bk)
        

        return g_bk * (v - self.e_bk)

    '''Calcium terms'''
    def r_ex(self, c):
        # [uM/s]
        return (c-self.c0)/self.tau_ex

    '''Stimulation'''
    def stim(self, t):
        if 1 <= t < 1.01 or 5 <= t < 5.01 or 9 <= t < 9.01 \
            or 13 <= t < 13.01 or 17 <= t < 17.01 or 21 <= t < 21.01 \
            or 25 <= t < 25.01 \
            or 30 <= t < 30.01 or 35 <= t < 35.01 or 40 <= t < 40.01 \
            or 45 <= t < 45.01 or 50 <= t < 50.01 or 55 <= t < 55.01 or 60 <= t < 60.01 \
            or 66 <= t < 66.01 or 72 <= t < 72.01:
            return 1
        else:
            return 0

    '''Numerical terms'''
    def rhs(self, y, t):
        # Right-hand side function
        c, v, n, hv, hc, x, z, p, q, bx, cx = y
        dcdt = - self.r_ex(c) \
            - 1e9 * self.i_cal(v, n, hv, hc) / (2 * self.F * self.d) \
            + 1e9 * self.i_cal(self.v0, self.n0, self.hv0, self.hc0) / (2 * self.F * self.d) \
            - 1e9 * self.i_cat(v, bx, cx) / (2 * self.F * self.d) \
            + 1e9 * self.i_cat(self.v0, self.bx0, self.cx0) / (2 * self.F * self.d)
        dvdt = - 1 / self.c_m * (self.i_cal(v, n, hv, hc) + self.i_cat(v, bx, cx) + self.i_kcnq(v, x, z) + self.i_kv(v, p, q) + self.i_kca(v, c) + self.i_bk(v) - 0.001 * self.stim(t))
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dxdt = (self.x_inf(v) - x)/self.tau_x(v)
        dzdt = (self.z_inf(v) - z)/self.tau_z(v)
        dpdt = (self.p_inf(v) - p)/self.tau_p(v)
        dqdt = (self.q_inf(v) - q)/self.tau_q(v)
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)

        return [dcdt, dvdt, dndt, dhvdt, dhcdt, dxdt, dzdt, dpdt, dqdt, dbxdt, dcxdt]

    def step(self):
        # Time stepping

        self.n0 = self.n_inf(self.v0)
        self.hv0 = self.hv_inf(self.v0)
        self.hc0 = self.hc_inf(self.c0)
        self.x0 = self.x_inf(self.v0)
        self.z0 = self.z_inf(self.v0)
        self.p0 = self.p_inf(self.v0)
        self.q0 = self.q_inf(self.v0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        y0 = [self.c0, self.v0, self.n0, self.hv0, self.hc0, self.x0, self.z0, self.p0, self.q0, self.bx0, self.cx0]
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

    '''Visualize results'''
    def plot(self, a, tmin=0, tmax=100, xlabel = 'time[s]', ylabel = None, color = 'b'):
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = FastCell(2)
    sol = model.step()
    c = sol[:, 0]
    v = sol[:, 1]
    n = sol[:, 2]
    hv = sol[:, 3]
    hc = sol[:, 4]
    x = sol[:, 5]
    z = sol[:, 6]
    p = sol[:, 7]
    q = sol[:, 8]
    bx = sol[:, 9]
    cx = sol[:, 10]

    # Plot the results
    plt.figure()
    plt.subplot(421)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(422)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(423)
    model.plot(model.i_cal(v, n , hv, hc), ylabel='i_cal[mA/cm^2]')
    plt.subplot(424)
    model.plot(model.i_cat(v, bx, cx), ylabel='i_cat[mA/cm^2]')
    plt.subplot(425)
    model.plot(model.i_kcnq(v, x, z), ylabel='i_kncq[mA/cm^2]')
    plt.subplot(426)
    model.plot(model.i_bk(v), ylabel='i_bk[mA/cm^2]')
    plt.subplot(427)
    model.plot(model.i_kca(v, c), ylabel = 'i_kca[mA/cm^2]')
    plt.subplot(428)
    model.plot([0.004 * model.stim(t) for t in model.time], ylabel='i_stim[mA/cm^2]', color = 'r')
    plt.show()