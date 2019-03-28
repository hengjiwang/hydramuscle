from parameters import *

# Terms on ER
def j_ip3r(c, c_t, h, ip):
    '''IP3R current, ER -> cytosol'''

    # global v_ip3r, gamma, d_1, d_5

    m_inf = ip/(ip+d_1)
    n_inf = c/(c+d_5)

    return v_ip3r * m_inf**2 * n_inf**2 * h**0.5 * \
    ((c_t-c)*gamma - c)**0.5

def j_leak(c, c_t):
    '''leak current, ER -> cytosol'''

    # global v_leak, gamma

    return v_leak * ((c_t-c)*gamma - c)

def j_serca(c):
    '''SERCA pump, cytosol -> ER'''

    # global v_serca, k_serca

    return v_serca* c**1.75 / (c**1.75 + k_serca**1.75)

# Terms on membrane
def j_in():
    '''input current, outside -> cytosol'''

    # global v_in

    return v_in

def j_out(c):
    '''output current, cytosol -> outside'''

    # global k_out

    return k_out * c

def j_pmca(c):
    '''PMCA pump, cytosol -> outside'''

    # global v_pmca, k_pmca

    return v_pmca * c**2 / (k_pmca**2 + c**2)

def j_soc(c, c_t):
    '''SOCC, outside -> cytosol'''

    # global v_soc, k_soc, gamma

    return v_soc * k_soc**4 / (k_soc**4 + ((c_t-c)*gamma)**4)

# Terms for IP3R
def h_inf(c, ip):
    '''stable IP3 inactivation rate'''

    # global d_2, d_1, d_3

    q_2 = d_2 * (ip + d_1)/(ip + d_3)
    return q_2 / (q_2 + c)

def tau_h(c, ip):
    '''time constant for IP3 inactivation rate'''

    # global d_2, d_1, d_3, a_2

    q_2 = d_2 * (ip + d_1)/(ip + d_3)
    return 1 / (a_2 * (q_2 + c))

# def j_stim(st):
#     '''stimulation-induced voltage increase'''
#     return v_st * st

def j_hynac(st):
    '''current through HyNaC'''
    return v_st * st

def j_k(v, n):
    '''current through potassium channel'''
    return g_k * n**4 * (v-e_k)

def alpha_n(v):
    '''rate constant for potassium channel'''
    return 0.01 * (v+55) / (1 - np.exp(-0.1 * (v+55)))

def beta_n(v):
    '''rate constant for potassium channel'''
    return 0.125 * np.exp(-0.0125 * (v+65))

def j_na(v, m, hh):
    '''current through sodium channel'''
    return g_na * m**3 * hh * (v - e_na)

def alpha_m(v):
    '''rate constant for sodium channel'''
    return 0.1 * (v+40)/(1 - np.exp(-0.1 * (v+40)))

def beta_m(v):
    '''rate constant for sodium channel'''
    return 4 * np.exp(-0.0556 * (v+65))

def alpha_hh(v):
    '''rate constant for sodium channel'''
    return 0.07 * np.exp(-0.05 * (v+65))

def beta_hh(v):
    '''rate constant for sodium channel'''
    return 1 / (1 + np.exp(-0.1 * (v+35)))

def j_l(v):
    '''voltage leak'''
    return v_l*(v-e_l)

def rhs22(y, t):
    '''right-hand side for integration in hypothesis 2'''
    c, c_t, h, ip, v, n, m, hh = y
    
    dcdt = j_ip3r(c, c_t, h, ip) - j_serca(c) + j_leak(c, c_t) + \
    (j_in() - j_out(c) - j_pmca(c) + j_soc(c,c_t))*delta
    
    dctdt =  (j_in() - j_out(c) - j_pmca(c) + j_soc(c,c_t))*delta
    
    dhdt = (h_inf(c, ip)-h)/tau_h(c, ip)
    
    dipdt = r_vip2 * (v-v0) - r_decay * ip
    
    dvdt = 1000*(j_hynac(stim(t)) - j_k(v, n)- j_na(v, m, hh) - j_l(v)) 

    dndt = alpha_n(v) * (1 - n) - beta_n(v) * n

    dmdt = alpha_m(v) * (1 - m) - beta_m(v) * m

    dhhdt = alpha_hh(v) * (1 - hh) - beta_hh(v) * hh
    
    return [dcdt, dctdt, dhdt, dipdt, dvdt, dndt, dmdt, dhhdt]

def stim(t):
    '''returns the stimulation status at time t'''
    if t >= st1 and t < st1+d_rise:
        return 1
    elif t >= st2 and t < st2+d_rise:
        return 0
    elif t >= st3 and t < st3+d_rise:
        return 0
    elif t >= st4 and t < st4+d_rise:
        return 0
    else:
        return 0

stim_train = np.array([stim(j*dt) for j in range(len(TIME))])
