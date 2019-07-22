from single_cell_models.single_cell_model_hh2_ip3st import SingleCellCalciumModelHH2
from fluorescence_encoder.fluorescence_encoder import FluorescenceEncoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.integrate import odeint
import pandas as pd

T = 20
dt = 0.001

class CellGridCalciumModel(SingleCellCalciumModelHH2):
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.gc = 500
        self.gip3 = 5
        onex = np.ones(self.num)
        oney = np.ones(self.num)
        
        Dx = spdiags(np.array([onex,-2*onex,onex]),\
        np.array([-1,0,1]),self.num,self.num).toarray()
        
        Dx[0,0] = -1
        Dx[self.num-1,self.num-1] = -1 
        
        Ix = np.eye(self.num)
        Dy = spdiags(np.array([oney,-2*oney,oney]),\
        np.array([-1,0,1]),self.num,self.num).toarray()
        Dy[0,0] = -1; Dy[self.num-1,self.num-1] = -1
        
        Iy = np.eye(self.num)
        self.L = np.kron(Dx, Iy) + np.kron(Ix, Dy)
        
        self.time = np.linspace(0, T, int(T/dt))
    
    # Override
    def rhs(self, y, t):
        
        c, c_t, hh, ip, v, m, h, n, m_cal, h_cal = y[0:self.num*self.num], \
        y[self.num*self.num:2*self.num*self.num], \
        y[2*self.num*self.num:3*self.num*self.num], \
        y[3*self.num*self.num:4*self.num*self.num], \
        y[4*self.num*self.num:5*self.num*self.num], \
        y[5*self.num*self.num:6*self.num*self.num], \
        y[6*self.num*self.num:7*self.num*self.num], \
        y[7*self.num*self.num:8*self.num*self.num], \
        y[8*self.num*self.num:9*self.num*self.num], \
        y[9*self.num*self.num:10*self.num*self.num]
    
        dcdt = (self.i_ip3r(c, c_t, hh, ip) \
             - self.i_serca(c) \
             + self.i_leak(c, c_t)) \
             + (- self.i_pmca(c) \
                - self.i_cal(v, m_cal, h_cal) \
                + self.i_soc(c, c_t) \
                - self.i_out(c)) * self.delta
        
        dctdt = (- self.i_pmca(c) \
                       - self.i_cal(v, m_cal, h_cal) \
                       + self.i_soc(c, c_t) \
                       - self.i_out(c))\
                 * self.delta

        dhhdt = (self.hh_inf(c, ip) - hh) / self.tau_hh(c, ip)

        dipdt = - self.ip_decay * (ip - self.ip0) + self.gip3 * self.L@ip
    
        dvdt = - (self.i_na(v,m,h) \
                  + self.i_k(v,n) \
                  + self.i_bk(v) \
                  + 2*self.i_cal(v, m_cal, h_cal))/self.c_m\
               + self.gc * self.L@v
        
        dipdt[-int(self.num/2)-1:-int(self.num/2)+2] += 2 * self.stim(t)
        dipdt[-int(self.num/2)-1 - self.num:-int(self.num/2)+2 - self.num] += 2 * self.stim(t)
        dipdt[-int(self.num/2)-1-2*self.num:-int(self.num/2)+2-2*self.num] += 2 * self.stim(t)
        
        dmdt = self.alpha_m(v) * (1-m) - self.beta_m(v) * m
        dhdt = self.alpha_h(v) * (1-h) - self.beta_h(v) * h
        dndt = self.alpha_n(v) * (1-n) - self.beta_n(v) * n
        dmcaldt = (self.m_cal_inf(v) - m_cal) / self.tau_cal_m(v)
        dhcaldt = (self.h_cal_inf(v) - h_cal) / self.tau_cal_h(v)
        
        deriv = np.array([dcdt, dctdt, \
                          dhhdt, dipdt, dvdt, dmdt, dhdt, dndt, dmcaldt, dhcaldt])
        
        dydt = np.reshape(deriv, 10*self.num*self.num)  
        
        return dydt
    
    # Override
    def stim(self, t):
        if t >= 1 and t < 1.1 or t >= 3 and t < 3.1 or t >= 5 and t < 5.1 or t >= 7 and t < 7.1 or t >= 9 and t < 9.1 or t >= 13 and t < 13.1: 
            return 0.5
        else:
            return 0
    
    # Override
    def step(self):
        self.hh0 = self.hh_inf(self.c0, self.ip0)
        
        y0 = np.array([self.c0*np.ones((self.num,self.num)), 
                       self.ct0*np.ones((self.num,self.num)), 
                       self.hh0*np.ones((self.num,self.num)), 
                       self.ip0*np.ones((self.num,self.num)), 
                       self.v0*np.ones((self.num,self.num)),
                       self.m0*np.ones((self.num,self.num)), 
                       self.h0*np.ones((self.num,self.num)), 
                       self.n0*np.ones((self.num,self.num)),
                       self.m_cal0*np.ones((self.num,self.num)),
                       self.h_cal0*np.ones((self.num,self.num))])

        y0 = np.reshape(y0, 10*self.num*self.num)
        
        sol = odeint(self.rhs, y0, self.time, hmax = 0.005)
        return sol

if __name__ == '__main__':
    n_cel = 20
    model = CellGridCalciumModel(n_cel)
    sol = model.step()
    c = np.reshape(sol[:,0:n_cel*n_cel], (-1,n_cel,n_cel))
    df = pd.DataFrame(np.reshape(c,(20000,400)))
    df.to_csv('save/data/c_2d_point_diff_20_3by3_pip5.csv', index = False)

