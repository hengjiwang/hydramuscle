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

    def plot(self, a, tmin=0, tmax=None, xlabel = 'time[s]', ylabel = None, color = 'b'):
        # Plot the time evolution of a
        tmax = self.T
        plt.plot(self.time[int(tmin/self.dt):int(tmax/self.dt)], a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)
