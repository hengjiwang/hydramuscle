import numpy as np
import matplotlib.pyplot as plt

class ReducedFastCell:

    def __init__(self, dt=0.001):
        self.dt = dt
        self.t = 0
        self.c0 = 0.05 # uM
        self.tau_stim = 0.05 # s
        self.r_stim = 1 # uM/s
        self.tau_inc = 0.075 # uM/s
        self.tau_dec = 0.5 # s
        self.c = self.c0
        self.c_train = [self.c]

        self.v0 = 0
        self.v_th = 0.1
        self.v_pk = 1.0
        self.v = self.v0
        self.c_m = 1
        self.r_m = 1
        self.tau_ref = 0.1 # s
        self.last_spike = -100 # s
        self.v_train = [self.v]

    def step(self, i_stim=0):
        # Update attributes

        if self.t - self.last_spike < self.tau_ref:
            # Cell is in refractory period
            self.v = self.v0
        else:
            self.v += self.dt / self.c_m * (i_stim - self.v / self.r_m)
            if self.v >= self.v_th:
                # Reach the threshold -- fire
                self.v = self.v_pk
                self.last_spike = self.t

        if self.t - self.last_spike < self.tau_stim:
            # Spike triggers calcium increasing
            self.c += self.dt / self.tau_inc

        self.c -= self.dt * (self.c - self.c0) / self.tau_dec
        self.c_train.append(self.c)
        self.v_train.append(self.v)
        self.t += self.dt

if __name__ == '__main__':
    model = ReducedFastCell()
    for t in np.arange(0, 2, 0.001):
        if 0 <= t <= 0.01 or 1 <= t <= 1.1:
            model.step(10)
        else:
            model.step(0)

    plt.figure()
    plt.plot(np.arange(0, 2.001, 0.001), model.v_train)
    plt.show()