import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABCMeta
from tqdm import tqdm

class CellBase(metaclass=ABCMeta):
    '''A base class for simulating single cell dynamics'''
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt)+1)
        self.c0=0.05

    @abstractmethod
    def rhs(self, y, t):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    # def euler_odeint(self, rhs, y, T, dt, save_interval=1, **kwargs):
    #     # An Euler integrator
    #     sol = np.zeros((int(T/dt/save_interval)+1, len(y)))

    #     for j in tqdm(np.arange(0, int(T/dt)+1)):
    #         t = j*dt
    #         dydt = rhs(y, t, **kwargs)
    #         y += dydt * dt
    #         if j % save_interval == 0:  sol[int(j/save_interval), :] = y

    #     return sol

    def plot(self, a, tmin=0, tmax=None, xlabel = 'time[s]', ylabel = None, color = 'b'):
        # Plot the time evolution of a
        tmax = self.T
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)
