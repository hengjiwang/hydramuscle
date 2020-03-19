import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABCMeta
from tqdm import tqdm

class CellBase(metaclass=ABCMeta):
    '''A base class for simulating single cell dynamics'''
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T-dt, int(T/dt))
        self.c0=0.05

    @abstractmethod
    def rhs(self, y, t):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
