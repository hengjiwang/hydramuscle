import numpy as np

# Numerical parameters
T = 50
dt = 0.001
TIME = np.linspace(0, T+dt, int(T/dt)+1)

# Initial conditions 
c0 = 0.08653896084067239
h0 = 0.6260398013478132
ct0 = 36.48955299437646
ip0 = 0
n0 = 0.3176806504981428
m0 = 0.052934144195247945
hh0 = 0.5960957200408543

# Volume ratio
gamma=5.4054

# Leak for ER
v_leak = 0.002

# Leak across plasma membrane
v_in = 0.05
k_out = 1.2

# IP3R parameters
v_ip3r = 0.4
d_1 = 0.13; d_2 = 1.049; d_3 = 0.9434; d_5= 0.08234; 
a_2 = 0.04

# PMCA terms
v_pmca = 10
k_pmca = 2.5

# SOC terms
v_soc = 1.57
k_soc = 90

# SERCA terms
v_serca = 0.9
k_serca = 0.1

# Sneyd Parameter
delta = 0.2

# IP3 parameters
d_rise = 0.005 #0.5
r_rise = 0.5 #0.025 
r_decay = 0.2

# Initial condition
v0 = -64.99973395350574

# j_stim terms
v_st = 50

# j_k terms
g_k = 36
e_k = -77

# j_na terms
g_na = 120
e_na = 50

# Voltage leak terms
v_l = 0.3
e_l = -54.4

# Voltage-dependent IP3 terms
r_vip2 = 0.2 #0.006

# Stimulation times
st1 = 20.0
st2 = 25.0
st3 = 29.0
st4 = 33.0