import numpy as np
from reduced_fast_cell import ReducedFastCell
from collections import defaultdict

class ReducedLayer:

    def __init__(self, dt=0.001, numx=30, numy=60):
        self.dt = dt
        self.numx = numx
        self.numy = numy
        self.layer = defaultdict(ReducedFastCell)
        self.stim_pattern = defaultdict(int)