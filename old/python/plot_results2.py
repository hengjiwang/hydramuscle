import matplotlib.pyplot as plt
from functions import *

def plot(TIME, c, c_t, ip, h, v):
    plt.figure()
    plt.subplot(121)
    plt.plot(TIME[0:-1], c, '-', label = 'c(t)')
    #plt.plot(TIME[6000:8000], c[6000:8000], '-', label = 'c(t)')
    #plt.plot(TIME[0:-1], ip, '--', label = 'ip(t)')
    plt.xlabel('time/s')
    plt.ylabel('concentration/uM')
    #plt.legend()
    # plt.show()

    plt.subplot(122)
    plt.plot(TIME[0:-1], v, label = 'membrane potential')
    plt.xlabel('time/s')
    plt.ylabel('voltage/mV')
    #plt.legend()
    plt.show()