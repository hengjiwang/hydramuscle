import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as numpy
from collections import defaultdict
from abc import abstractmethod, ABCMeta

class PopBase(metaclass=ABCMeta):
    """An abstract base class for simulating multicellular dynamics"""
    def __init__(self, cell, save_interval):
        self.cell = cell
        self.T = cell.T
        self.dt = cell.dt
        # self._gc = None
        # self._gip3 = None
        # self._num = None
        self._stims_v_map = defaultdict(list)
        self._stims_ip_map = defaultdict(list)
        self._save_interval = save_interval
    
    
    @abstractmethod
    def _set_conn_mat(self):
        "Set the connetivity matrix"
        raise NotImplementedError

    @abstractmethod
    def set_stim_pattern(self, pathway,
                         xmin, xmax, ymin, ymax,
                         stim_times, randomnum=0):
        "Set the stimulation pattern"
        raise NotImplementedError

    @abstractmethod
    def _rhs(self, y, t):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError