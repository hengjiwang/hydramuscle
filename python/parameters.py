import numpy as np
from scipy.sparse import spdiags

# Numerical parameters
T = 50
dt = 0.0001
TIME = np.linspace(0, T+dt, int(T/dt)+1)

# Initial conditions 
c0 = 0.08653896084067239
h0 = 0.20273225631266437 # 0.6260398013478132
ct0 = 36.48955299437646
ip0 = 0.01 # 0
n0 = 0.3176806504981428
m0 = 0.052934144195247945
hh0 = 0.5960957200408543

# Volume ratio
gamma=5.4054

# Leak for ER
v_leak = 0.002

# Leak across plasma membrane
v_in =  0.05 -0.39473779866524883 #-0.04762511540896227
k_out = 1.2

# IP3R parameters
v_ip3r = 5 # 0.4
d_1 = 0.01 # 0.13 
d_2 = 1.049; d_3 = 0.9434; d_5= 0.08234; 
a_2 = 0.04

# PMCA terms
v_pmca = 10
k_pmca = 2.5

# SOC terms
v_soc = 1.57
k_soc = 90

# SERCA terms
v_serca = 1.44705290611436 # 1.8206604887397753# 1.5137717346418553 # 0.9
k_serca = 0.1

# Sneyd Parameter
delta = 0.2

# IP3 parameters
d_rise = 0.05 #0.5
r_rise = 0.5 #0.025 
r_decay = 0.5 # 0.2

# Initial condition
v0 = -64.99973395350574

# j_hynac terms
v_st = 80 #50

# j_k terms
g_k = 36
e_k = -77

# j_na terms
g_na = 120
e_na = 50

# Voltage leak terms
v_l = 0.3
e_l = -57.03158532907785 # -54.4 -2.6315853244349925 #-0.3175007693930818

# Temperature parameter
Temp = 18
phi = 3 ** ((Temp - 6.3)/10)

# # Voltage-dependent IP3 terms
# r_vip2 = 0.2 #0.006

# VGCC terms
g_ca = 0.3 # 0.036195
v_ca1 = 100
v_ca2 = -24
r_ca = 8.5

# PLCd terms
v_7 = 0
k_ca = 0.2 # 0.3

# Stimulation times
st1 = 20.0
st2 = 23.0
st3 = 26.0
st4 = 29.0
st5 = 31.5


#### parameters for multicellular model

# Number of cells
N = 10

# Electrical coupling coefficient
gc = 1000

# IP3 diffusion coeffcient
gip = 100

# Grid initialization
vec_c = np.zeros(N)
vec_ct = np.zeros(N)
vec_h = np.zeros(N)
vec_ip = np.zeros(N)
vec_v = np.zeros(N)

# coupling matrix
onex = np.ones(N)
oney = np.ones(N)
Dx = spdiags(np.array([onex,-2*onex,onex]),\
        np.array([-1,0,1]),N,N).toarray()
Dx[0,0] = -1; Dx[N-1,N-1] = -1
Ix = np.eye(N)
Dy = spdiags(np.array([oney,-2*oney,oney]),\
        np.array([-1,0,1]),N,N).toarray()
Dy[0,0] = -1; Dy[N-1,N-1] = -1
Iy = np.eye(N)
L = np.kron(Dx, Iy) + np.kron(Ix, Dy)

