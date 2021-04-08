import numpy as np
from hydramuscle.reduced_model.reduced_fast_cell import ReducedFastCell

class ReducedLayer:

    def __init__(self, dt=0.001, numx=30, numy=60, gc=100, stim_amp=10):
        self.dt = dt
        self.numx = numx
        self.numy = numy
        self.gc = gc
        self.stim_amp = stim_amp
        self.set_conn_pattern()
        self.v_last = np.zeros((self.numx, self.numy)) + self.layer[0,0].v0

    def set_conn_pattern(self):

        # Initiate layer
        self.layer = np.array([[None] * self.numy for _ in range(self.numx)])
        for x in range(self.numx):
            for y in range(self.numy):
                self.layer[x, y] = ReducedFastCell(self.dt)

    def step(self, stim_pattern=set()):
        # Update cells
        for x in range(self.numx):
            for y in range(self.numy):

                i_stim = 0

                if (x, y) in stim_pattern:
                    i_stim += self.stim_amp

                # Couple neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    x2 = x + dx
                    y2 = y + dy

                    if x2 == self.numx:
                        x2 = 0
                    elif x2 == -1:
                        x2 = self.numx - 1

                    if 0 <= y2 <= self.numy - 1:
                        i_stim += self.gc * (self.v_last[x2, y2] - self.v_last[x, y])

                self.layer[x, y].step(i_stim)

        # Update v_last
        for x in range(self.numx):
            for y in range(self.numy):
                self.v_last[x, y] = self.layer[x, y].v











