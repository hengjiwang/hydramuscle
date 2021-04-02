import numpy as np
from hydramuscle.reduced_model.reduced_fast_cell import ReducedFastCell

class ReducedLayer:

    def __init__(self, dt=0.001, numx=30, numy=60):
        self.dt = dt
        self.numx = numx
        self.numy = numy
        self.set_conn_pattern()

    def set_conn_pattern(self):

        # Initiate layer
        self.layer = np.array([[None] * self.numy for _ in range(self.numx)])
        for x in range(self.numx):
            for y in range(self.numy):
                self.layer[x, y] = ReducedFastCell(self.dt)

        # Set connections
        for x in range(self.numx):
            for y in range(self.numy):

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    x2 = x + dx
                    y2 = y + dy

                    if x2 == self.numx:
                        x2 = 0
                    elif x2 == -1:
                        x2 = self.numx - 1

                    if 0 <= y2 < self.numy:
                        self.layer[x, y].neighbors.append(self.layer[x2, y2])

                # print(len(self.layer[0,0].neighbors))

    def step(self, stim_pattern=set()):

        # Step directly stimulated cells
        for (x, y) in stim_pattern:
            cell = self.layer[x, y]
            cell.step(stim=True)

        # Step other cells
        for x in range(self.numx):
            for y in range(self.numy):
                if (x, y) not in stim_pattern:
                    cell = self.layer[x, y]
                    cell.step()









