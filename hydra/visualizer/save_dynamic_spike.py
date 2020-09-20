import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

init1 = 0
init2 = 123
stims_fast = [init1+0.0, init1+9.25, init1+14.5, init1+18.25, init1+21.5, init1+24.5, init1+28.0, init1+31.5, init1+35.25, init1+40.5,
              init2+0.0, init2+7.75, init2+15.25, init2+19.75, init2+23.0, init2+27.25, init2+30.5, init2+34.25, init2+38.25, init2+43.5]
stims_slow = [init1+15, init2+15]

# DPI = 200
plt.ion()
# plt.figure(figsize=(1280/DPI, 330/DPI), dpi=DPI)
plt.figure(1)

t= [0]
t_now = 0
m1 = []
m2 = []

for i in range(2500):

    print(i)

    plt.clf()
    t_now = i * 0.1
    t.append(t_now)
#     m1.append(t_now in stims_fast)
#     m2.append(t_now in stims_slow)
    
    m1.append(False)
    for x in stims_fast:
        if 0 <= t_now - x <= 1:
            m1[-1] = True
            break
            
    m2.append(False)
    for x in stims_slow:
        if 0 <= t_now - x <= 4:
            m2[-1] = True
            break
    
    for moment in m1:
        if moment:
            plt.vlines(moment, 0, 1, color='b', linewidth=0.1)
            
    for moment in m2:
        if moment:
            plt.vlines(moment, 0, 1, color='r', linewidth=0.1, alpha=0.5)
    
    plt.xlim(-5, 250)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel('time(s)')
    # plt.draw()
    # time.sleep(0.01)
    plt.pause(0.01)