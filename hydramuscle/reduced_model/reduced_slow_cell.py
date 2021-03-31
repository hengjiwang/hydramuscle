import numpy as np
import matplotlib.pyplot as plt

class ReducedSlowCell:

    def __init__(self, dt=0.001):
        self.dt = dt
        self.t = 0 # s
        self.c0 = 0.05 # uM
        self.ip_thres = 0.3 # uM
        self.c = self.c0
        self.c_train = [self.c]
        self.r_dec = 0.5 # uM/s
        self.r_rel = 1 # uM/s

    def step(self, ip):
        self.c += self.dt * (self.r_rel * (ip > self.ip_thres)  - self.r_dec * (self.c - self.c0))
        self.c_train.append(self.c)
        self.t += self.dt


if __name__ == '__main__':

    model = ReducedSlowCell()

    ip_train = [0.01]
    for t in np.arange(0, 100, 0.001):
        ip = ip_train[-1]
        if t < 4:
            ip_train.append(ip + 0.001 * (1 - 0.2 * ip))
        else:
            ip_train.append(ip + 0.001 * - 0.2 * ip)
        model.step(ip)

    plt.figure()
    plt.plot(np.arange(0, 100.001, 0.001), ip_train)
    plt.plot(np.arange(0, 100.001, 0.001), model.c_train)
    plt.show()



