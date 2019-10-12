#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.sparse import spdiag
import scipy
import pandas as pd
from numba import jitclass, int32, float64
import os

spec = [('k1', float64),
('k2', float64),
('ka', float64),
('kip', float64),
('k3', float64),
('v40', float64),
('v41', float64),
('kr', float64),
('k5', float64),
('k6', float64),
('ki', float64),
('kg', float64),
('a0', float64),
('v7', float64),
('v8', float64),
('kca', float64),
('k9', float64),
('beta', float64),
('c0', float64),
('s0', float64),
('r0', float64),
('ip0', float64),
('T', float64),
('dt', float64),
('time', float64[:]),
('c_m', float64),
('A_cyt', float64),
('V_cyt', float64),
('d', float64),
('F', float64),
('v0', float64),
('n0', float64),
('hv0', float64),
('hc0', float64),
('bx0', float64),
('cx0', float64),
('g_cal', float64),
('e_cal', float64),
('k_cal', float64),
('g_cat', float64),
('e_cat', float64),
('g_kcnq', float64),
('e_k', float64),
('g_kv', float64),
('g_kca', float64),
('g_bk', float64),
('e_bk', float64),
('gc', float64),
('g_ip3', float64),
('num', int32),
('num2', int32),
('L', float64[:,:]),
('scale_stim_v', float64),
('scale_stim_ip', float64),
('r_inc', float64),
('k1p', float64),
('k1n', float64),
('k2p', float64),
('k2n', float64),
('k3p', float64),
('k3n', float64),
('k4p', float64),
('k4n', float64),
('g0', float64),
('c1g0', float64),
('c2g0', float64),
('c3g0', float64),
('c4g0', float64),
('phi0', float64),
('phi1', float64),
('phi2', float64),
('phi3', float64),
('phi4', float64),
('const_iin', float64)]

num = 20

onex = np.ones(num)
Dx = spdiags(np.array([onex,-2*onex,onex]), np.array([-1,0,1]),num,num).toarray()
Dx[0,0] = -1
Dx[num-1,num-1] = -1 
Ix = np.eye(num)
L = np.kron(Dx, Ix) + np.kron(Ix, Dx)

# @jitclass(spec)
class Grid():
    '''A 2D dynamical model with cells connected by gap junctions'''
    def __init__(self, num=num, T=300, dt = 0.001, k2=0.2, s0=600, d=10e-4, v7=0, k9=0.01, scale_stim_v = 0.01, scale_stim_ip = 1.0, L=L):
        
        # General parameters
        self.gc = 5e4
        self.g_ip3 = 1

        # Hofer parameters
        self.k2 = 0.08
        self.ka = 0.2
        self.kip = 0.3
        self.k3 = 0.5
        self.v40 = 0.025
        self.v41 = 0.2
        self.kr = 1
        self.k5 = 0.5
        self.k6 = 4
        self.ki = 0.2
        self.kg = 0.1 # unknown
        self.a0 = 1 # 1e-3 - 10
        self.v7 = 0.04 # 0 - 0.05
        self.v8 = 0.00012 # 4e-4
        self.kca = 0.3
        self.k9 = 0.08
        self.beta = 20

        self.c0 = 0.05
        self.s0 = 60
        self.r0 = 0.9411764705882353
        self.ip0 = 0.01

        # Fast Parameters
        self.c_m = 1e-6 # [F/cm^2]
        self.A_cyt = 4e-5 # [cm^2]
        self.V_cyt = 6e-9 # [cm^3]
        self.d = 10e-4 # 10e-4 # [cm]
        self.F = 96485332.9 # [mA*s/mol]
        self.v0 = -50 # (-40 to -60)
        self.n0 = 0.00591106885624379
        self.hv0 = 0.8232409668812207
        self.hc0 = 0.9523809523809523
        self.bx0 = 0.06951244510501192
        self.cx0 = 0.06889595335007676

        # Fluorescence Parameters
        self.r_inc = 200
        self.k1p = 2.5
        self.k1n = 0.1
        self.k2p = 16.9
        self.k2n = 205
        self.k3p = 1.1
        self.k3n = 11.8
        self.k4p = 1069
        self.k4n = 5.8
        self.g0 = 3.281
        self.c1g0 = 4.101
        self.c2g0 = 0.017
        self.c3g0 = 0
        self.c4g0 = 0.0007
        self.phi0 = 1
        self.phi1 = 1
        self.phi2 = 1
        self.phi3 = 1
        self.phi4 = 81
        
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
        self.e_bk = -55


        # Build grid
        self.num = num
        self.num2 = self.num ** 2
        self.L = L

        # Modified parameters
        # self.ip_decay = 0.04
        self.k9 = k9 
        self.d = d 
        self.v7 = v7 
        self.k2 = k2 
        self.s0 = s0
        self.scale_stim_v = scale_stim_v
        self.scale_stim_ip = scale_stim_ip

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, 1+int(T/dt))

        self.g_bk = 2.028567365131953e-05
        self.const_iin = -0.5049104594276217
        self.k1 = 2.938083356243623e-05

    '''Hofer methods'''
    def i_rel(self, c, s, ip, r):
        # Release from ER, including IP3R and leak term [uM/s]
        return (self.k2 * r * c**2 * ip**2 / (self.ka**2 + c**2) / (self.kip**2 + ip**2)) * (s - c)

    def i_serca(self, c):
        # SERCA [uM/s]
        # v_serca = 2
        # k_serca = 0.1
        # return v_serca * c / (c + k_serca)
        return self.k3 * c

    def i_leak(self, c, s):
        return self.k1 * (s - c)

    def i_in(self, ip):
        return self.const_iin + self.v41 * ip**2 / (self.kr**2 + ip**2)

    def i_out(self, c):
        # Additional eflux [uM/s]
        return self.k5 * c

    def v_r(self, c, r):
        # Rates of receptor inactivation and recovery [1/s]
        return self.k6 * (self.ki**2 / (self.ki**2 + c**2) - r)

    def i_plcb(self, v8):
        # Agonist-controlled PLC-beta activity [uM/s]
        return v8 * 1 / ((1 + self.kg)*(self.kg/(1+self.kg) + self.a0)) * self.a0

    def i_plcd(self, c):
        # PLC-delta activity [uM/s]
        return self.v7 * c**2 / (self.kca**2 + c**2)

    def i_deg(self, ip):
        # IP3 degradion [uM/s]
        return self.k9 * ip

    def stim(self, t, stims):
        # Stimulation

        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + 4

        return int(condition)

    '''Fast methods'''
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

    def i_kca(self, v, c):
        return self.g_kca * 1 / (1 + np.exp(v/(-17) - 2 * np.log(c))) * (v - self.e_k)

    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]

        return self.g_bk * (v - self.e_bk)

    def stim_v(self, t, stims):

        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + 0.01

       	return int(condition)

    '''Fluorescence methods'''
    def f_total(self, g, c1g, c2g, c3g, c4g):
        f_cyt = self.phi0*g + self.phi1*c1g + self.phi2*c2g + \
        self.phi3*c3g + self.phi4*c4g
        f_cyt0 = self.phi0*self.g0 + self.phi1*self.c1g0 + \
        self.phi2*self.c2g0 + self.phi3*self.c3g0 + self.phi4*self.c4g0
        f_bg = f_cyt0 * 2.5
        return (f_cyt + f_bg) / (f_cyt0 + f_bg)
    
    def r_1(self, c, g, c1g):
        return self.k1p * c * g - self.k1n * c1g
    
    def r_2(self, c, c1g, c2g):
        return self.k2p * c * c1g - self.k2n * c2g
    
    def r_3(self, c, c2g, c3g):
        return self.k3p * c * c2g - self.k3n * c3g

    def r_4(self, c, c3g, c4g):
        return self.k4p * c * c3g - self.k4n * c4g

    '''Time stepping'''
    def rhs(self, y, t, stims_v, stims_ip):
        # Right-hand side formulation

        # Assign variables
        num = self.num
        num2 = self.num2
        c, s, r, ip, v, n, hv, hc, bx, cx, g, c1g, c2g, c3g, c4g = (y[0:num2], 
        y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2], y[4*num2:5*num2], 
        y[5*num2:6*num2], y[6*num2:7*num2], y[7*num2:8*num2], y[8*num2:9*num2], 
        y[9*num2:10*num2], y[10*num2:11*num2], y[11*num2:12*num2], y[12*num2:13*num2],
        y[13*num2:14*num2], y[14*num2:15*num2])

        # Current terms which will be used for multiple times
        irel = self.i_rel(c, s, ip, r)
        ileak = self.i_leak(c, s)
        iserca = self.i_serca(c)
        iin = self.i_in(ip)
        iout = self.i_out(c)
        ical = self.i_cal(v, n, hv, hc)
        icat = self.i_cat(v, bx, cx)
        iplcb_stim = self.scale_stim_ip * self.i_plcb(self.stim(t, stims_ip))
        iplcb_rest = self.i_plcb(self.v8)
        ir1 = self.r_1(c, g, c1g)
        ir2 = self.r_2(c, c1g, c2g)
        ir3 = self.r_3(c, c2g, c3g)
        ir4 = self.r_4(c, c3g, c4g)

        # Cytosolic calcium
        dcdt = irel + ileak - iserca + iin - iout - 1e9 * (ical + icat) / (2 * self.F * self.d) \
            - ir1 - ir2 - ir3 - ir4

        # Total calcium
        dsdt = self.beta * (iserca - irel - ileak)
        
        # Inactivation rate of IP3R
        drdt = self.v_r(c, r)

        # IP3 of downstream cells
        dipdt = iplcb_rest + self.i_plcd(c) - self.i_deg(ip) + self.g_ip3 * self.L@ip
        
        # IP3 of stimulated cells
        dipdt[-int(num/2)-1         : -int(num/2) + 2        ] += iplcb_stim - iplcb_rest
        dipdt[-int(num/2)-1 -   num : -int(num/2) + 2 -   num] += iplcb_stim - iplcb_rest
        dipdt[-int(num/2)-1 - 2*num : -int(num/2) + 2 - 2*num] += iplcb_stim - iplcb_rest
    
        # Voltage of downstream cells
        dvdt = - 1 / self.c_m * (ical + icat + self.i_kca(v, c) + self.i_bk(v)) + self.gc * self.L@v
        
        # Voltage of stimulated cells
        dvdt[0:3*num] += 1 / self.c_m * self.scale_stim_v * self.stim_v(t, stims_v)

        # CaL and K channel factors
        dndt = (self.n_inf(v) - n)/self.tau_n(v)
        dhvdt = (self.hv_inf(v) - hv)/self.tau_hv(v)
        dhcdt = (self.hc_inf(c) - hc)/self.tau_hc()
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)
        dgdt = - ir1
        dc1gdt = (ir1 - ir2)
        dc2gdt = (ir2 - ir3)
        dc3gdt = (ir3 - ir4)
        dc4gdt = ir4

        # Put together
        dydt = np.concatenate((dcdt, dsdt, drdt, dipdt, dvdt, dndt, dhvdt, dhcdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt)) 

        return dydt

def step(model, stims_v = [201,203,205,207,209,211,213,215,217,219], stims_ip = [10]):
    # Time stepping

    start_time = time.time() # Begin counting time
    
    base_mat = np.ones((model.num,model.num))

    inits = [model.c0, model.s0, model.r0, model.ip0, model.v0, model.n0, model.hv0, 
    model.hc0, model.bx0, model.cx0, model.g0, model.c1g0, model.c2g0, model.c3g0, model.c4g0]

    y0 = np.array([x*base_mat for x in inits])

    y0 = np.reshape(y0, 15*model.num2)   

    sol = odeint(model.rhs, y0, model.time, args = (np.array(stims_v), np.array(stims_ip)), hmax = 0.1)

    elapsed = (time.time() - start_time) # End counting time
    print("Num: " + str(model.num) + "; Time used:" + str(elapsed))

    return sol

if __name__ == '__main__':
    n_cel = num
    model = Grid(n_cel, 100, 0.1)
    sol = step(model)
    # c = np.reshape(sol[:,0:n_cel*n_cel], (-1,n_cel,n_cel))
    # df = pd.DataFrame(np.reshape(c,(-1,n_cel**2)))
    df = pd.DataFrame(sol[:,0:n_cel*n_cel])
    df.to_csv('c_20x20_100s.csv', index = False)