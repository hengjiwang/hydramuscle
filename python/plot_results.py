import matplotlib.pyplot as plt
from functions import *

def plot(TIME, c, c_t, ip, h, v):
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(TIME[0:-1], c, 'k-', label = 'c(t)')
    plt.plot(TIME[0:-1], ip, 'r--', label = 'ip(t)')
    plt.xlabel('time/s')
    plt.ylabel('concentration/uM')
    plt.legend()
    #plt.show()

    plt.subplot(2,2,2)
    #plt.plot(TIME[0:-1], c_t, 'b-', label = 'c_t(t)')
    plt.plot(TIME[0:-1], (c_t-c)*gamma, 'g-', label = 'c_ER(t)')
    plt.xlabel('time/s')
    #plt.ylabel('concentration/uM')
    plt.legend()
    #plt.show()

    plt.subplot(2,2,3)
    plt.plot(TIME[0:-1], j_ip3r(c, c_t, h, ip), 'r-', label = 'j_ip3r')
    plt.plot(TIME[0:-1], -j_serca(c), 'g-', label = 'j_serca')
    plt.plot(TIME[0:-1], j_leak(c,c_t), 'b-', label = 'j_leak')
    plt.xlabel('time/s')
    plt.ylabel('current[uM/s]')
    plt.legend(loc='upper right')
    #plt.show()

    plt.subplot(2,2,4)
    plt.plot(TIME[0:-1], [j_in()]*(len(TIME)-1), 'k--', label = 'j_in')
    plt.plot(TIME[0:-1], -j_out(c), 'r-', label = 'j_out')
    plt.plot(TIME[0:-1], -j_pmca(c), 'g-', label = 'j_pmca')
    plt.plot(TIME[0:-1], j_soc(c,c_t), 'b-', label = 'j_soc')
    plt.xlabel('time/s')
    # plt.ylabel('current/uM')
    plt.legend(loc='upper right')

    plt.show()

    plt.figure()
    plt.plot(TIME[0:-1], v, 'k-', label = 'membrane potential')
    plt.xlabel('time/s')
    plt.ylabel('voltage/mV')
    plt.legend()
    plt.show()