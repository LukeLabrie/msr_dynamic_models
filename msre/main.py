from parameters import *
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

def dydtMSRE(t,y,delays,rho_ext):

    '''
    Returns derivative of state vector y; y' or dy/dt, of the MSRE system.

    The y vector contains the following:

        T_in_rc = Inlet temperature (°C) of radiator coolant, will be equal to 
                  the outlet temperature of the heat exchanger (T_hc4) plus 
                  relevant time delay

        T_out_rc = Outlet temperature (°C) of radiator coolant

        T_in_air = Inlet temperature (°C) of air in radiator

        T_out_air = Outlet temperature (°C) of air in radiator

        T_in_hf = Inlet temperature (°C) of heat exchanger fuel, will be equal 
                  to the outlet temperature of the core (T_cf2) plus relevant 
                  time delay

        T_hf* = Temperature (°C) of heat exchanger fuel node *

        T_ht* = Temperature (°C) of heat exchanger tube node *

        T_hc* = Temperature (°C) of heat exchanger coolant node *

        T_in_cf = Inlet temperature (°C) of core fuel, will be equal to the 
                  outlet temperature of the heat exchanger (T_hf4) plus 
                  relevant time dela

        S = neutro source perturbation term 

        rho_fb = feedback reactivity (from fuel and graphite temperatures)

        rho_ext = external reactivity (reactivity insertion)

        rho_tot = total reactivity = rho_0 + rho_fb + rho_ext (rho_0 = 
                  steady-state reactivity, constant)

        n = neutron density n(t)

        C* = precursor concentration of group *

        T_cg = Temperature (°C) of core graphite node 

        T_cf* = Temperature (°C) of core fuel node *  

        *_delay = Parameter at time t = t - delay

    Other parameters are defined in parameters.py
    '''

    # unpack state variables 
    T_in_rc, T_out_rc, T_out_air,T_in_hf, T_hf1, T_hf2, T_hf3, T_hf4, T_ht1, \
    T_ht2, T_in_hc, T_hc1, T_hc2, T_hc3, T_hc4, T_in_cf, n, C1, C2,    \
    C3, C4, C5, C6, T_cg, T_cf1, T_cf2 = y
    
    # delay terms
    T_out_rc_delay, T_out_air_delay, T_in_rc_delay, T_hc3_delay, T_hc4_delay, \
    T_ht1_delay, T_ht2_delay, T_cf1_delay, T_cf2_delay, T_cg_delay, n_delay,  \
    T_hf3_delay, T_hf4_delay, C1_delay, C2_delay, C3_delay, C4_delay,         \
    C5_delay, C6_delay = delays
    # derivatives 

    # need to handle time delays, perhaps take one-step at a time, update y_0
    # or use function as input which calculates necessary delay terms 

    rho = (a_f/2)*((-T0_f1+T_cf1)+(-T0_f2+T_cf2)) + a_g*(-T0_g1+T_cg) + rho_ext

    # radiator coolant outlet 
    if (t > tau_r_hx):
        dTdt_hc_in = -((W_rp/mn_rp)+(hA_rpn/mcp_rpn))*T_out_rc_delay + (hA_rpn/mcp_rpn)*T_out_air_delay + (W_rp/mn_rp)*T_in_rc_delay # T_in_hc            
    else:
        dTdt_hc_in = 0.0

    # radiator coolant inlet 
    if (t > tau_hx_r):
        dTdt_rc_in = -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc4_delay + (hA_sn/mcp_sn)*T_ht1_delay + (W_s/mn_s)*T_hc3_delay                   # T_in_rc            
    else:
        dTdt_rc_in = 0.0

    # core fuel inlet 
    if (t > tau_hx_c):
        dTdt_cf_in = -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf4_delay + (hA_pn/mcp_pn)*T_ht2_delay + (W_p/mn_p)*T_hf3_delay                    # T_in_cf
    else:
        dTdt_cf_in = 0.0 

    # hx fuel inlet 
    if (t > tau_c_hx):
        dTdt_hf_in = -((W_f/mn_f)+(hA_fg*k_2/mcp_f2))*T_cf2_delay + (hA_fg*k_2/mcp_f2)*T_cg_delay + (W_f/mn_f)*T_cf1_delay + (k_f2*P/mcp_f2)*n_delay # T_in_hf
    else:
        dTdt_hf_in = 0.0

    # hx precursors 
    if (t > tau_l):
        dC1dt = n*beta[0]/Lam-lam[0]*C1-C1/tau_c+C1_delay*np.exp(-lam[0]*tau_l)/tau_c  # C1
        dC2dt = n*beta[1]/Lam-lam[1]*C2-C2/tau_c+C2_delay*np.exp(-lam[1]*tau_l)/tau_c  # C2
        dC3dt = n*beta[2]/Lam-lam[2]*C3-C3/tau_c+C3_delay*np.exp(-lam[2]*tau_l)/tau_c  # C3
        dC4dt = n*beta[3]/Lam-lam[3]*C4-C4/tau_c+C4_delay*np.exp(-lam[3]*tau_l)/tau_c  # C4
        dC5dt = n*beta[4]/Lam-lam[4]*C5-C5/tau_c+C5_delay*np.exp(-lam[4]*tau_l)/tau_c  # C5
        dC6dt = n*beta[5]/Lam-lam[5]*C6-C6/tau_c+C6_delay*np.exp(-lam[5]*tau_l)/tau_c  # C6
    else:
        dC1dt = n*beta[0]/Lam-lam[0]*C1-C1/tau_c+C1*np.exp(-lam[0]*tau_l)/tau_c  # C1
        dC2dt = n*beta[1]/Lam-lam[1]*C2-C2/tau_c+C2*np.exp(-lam[1]*tau_l)/tau_c  # C2
        dC3dt = n*beta[2]/Lam-lam[2]*C3-C3/tau_c+C3*np.exp(-lam[2]*tau_l)/tau_c  # C3
        dC4dt = n*beta[3]/Lam-lam[3]*C4-C4/tau_c+C4*np.exp(-lam[3]*tau_l)/tau_c  # C4
        dC5dt = n*beta[4]/Lam-lam[4]*C5-C5/tau_c+C5*np.exp(-lam[4]*tau_l)/tau_c  # C5
        dC6dt = n*beta[5]/Lam-lam[5]*C6-C6/tau_c+C6*np.exp(-lam[5]*tau_l)/tau_c  # C6

    dydt = [ 
    dTdt_rc_in,                                                                                      # T_in_rc
    -((W_rp/mn_rp)+(hA_rpn/mcp_rpn))*T_out_rc + (hA_rpn/mcp_rpn)*T_out_air + (W_rp/mn_rp)*T_in_rc ,  # T_out_rc                                                                                   # T_out_rc
    -((W_rs/mn_rs)+(hA_rsn/mcp_rsn))*T_out_air + (hA_rsn/mcp_rsn)*T_out_rc + (W_rs/mn_rs)*Trs_in,    # T_out_air
    dTdt_hf_in,                                                                                      # T_in_hf
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf1 + (hA_pn/mcp_pn)*T_ht1 + (W_p/mn_p)*T_in_hf,                  # T_hf1
    (W_p/mn_p)*(T_hf1-T_hf2) + (hA_pn/mcp_pn)*(T_ht1-T_hf1),                                         # T_hf2
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf3 + (hA_pn/mcp_pn)*T_ht2 + (W_p/mn_p)*T_hf2,                    # T_hf3
    (W_p/mn_p)*(T_hf3-T_hf4) + (hA_pn/mcp_pn)*(T_ht2-T_hf3),                                         # T_hf4
    -2*((hA_pn/mcp_tn)+(hA_sn/mcp_tn))*T_ht1 + (2*hA_pn/mcp_tn)*T_hf1 + (2*hA_sn/mcp_tn)*T_hc3,      # T_ht1
    (2*hA_pn/mcp_tn)*(T_hf3-T_ht2) + (2*hA_sn/mcp_tn)*(T_hc1-T_ht2),      # T_ht2
    dTdt_hc_in,                                                                                      # T_in_hc
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc1 + (hA_sn/mcp_sn)*T_ht2 + (W_s/mn_s)*T_in_hc,                  # T_hc1
    (W_s/mn_s)*(T_hc1-T_hc2) + (hA_sn/mcp_sn)*(T_ht2-T_hc1),                                         # T_hc2
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc3 + (hA_sn/mcp_sn)*T_ht1 + (W_s/mn_s)*T_hc2,                    # T_hc3
    (W_s/mn_s)*(T_hc3-T_hc4) + (hA_sn/mcp_sn)*(T_ht1-T_hc3),                                         # T_hc4
    dTdt_cf_in,                                                                                      # T_in_cf
    (rho-beta_t)*n/Lam+lam[0]*C1+lam[1]*C2+lam[2]*C3+lam[3]*C4+lam[4]*C5+lam[5]*C6,                  # n (no source insertion)
    dC1dt,                                                                                           # C1
    dC2dt,                                                                                           # C2
    dC3dt,                                                                                           # C3
    dC4dt,                                                                                           # C4
    dC5dt,                                                                                           # C5
    dC6dt,                                                                                           # C6
    (hA_fg/mcp_g1)*(T_cf1 - T_cg) + k_g*P*n/mcp_g1,                                                  # T_cg
    W_f/mn_f*(T_in_cf - T_cf1) + (k_f1*P*n/mcp_f1) + (hA_fg*k_1*(T_cg - T_cf1)/mcp_f1),              # T_cf1   
    W_f/mn_f*(T_cf1 - T_cf2) + (k_f2*P*n/mcp_f2) + (hA_fg*k_2*(T_cg - T_cf1)/mcp_f2)                 # T_cf2   
    ]
    return dydt

def get_delay_indicies(t_current,timeVec):

    # indicies of delay
    d_indicies = [0]*19

    # radiator coolant outlet 
    if (t_current > tau_r_hx):

        # delay time
        t_d = t_current-tau_r_hx

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_current < diff_min):
                i_min = t[0]
                diff_min = t[1]-t_d
        
        d_indicies[0] = i_min # radiator coolant outlet
        d_indicies[1] = i_min # radiator air outlet 
        d_indicies[2] = i_min # radiator coolant inlet 

    # radiator coolant inlet 
    if (t_current > tau_hx_r):

        # delay time
        t_d = t_current-tau_hx_r

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_current < diff_min):
                i_min = t[0]
                diff_min = t[1]-t_d
        
        d_indicies[3] = i_min # hx coolant outlet (coolant node 3)
        d_indicies[4] = i_min # hx coolant outlet (coolant node 4)
        d_indicies[5] = i_min # hx coolant outlet (tube node 1)

    # core fuel inlet 
    if (t_current > tau_hx_c):

        # delay time
        t_d = t_current-tau_hx_c

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_current < diff_min):
                i_min = t[0]
                diff_min = t[1]-t_d
        
        d_indicies[6] = i_min # hx fuel outlet (tube node 2)
        d_indicies[11] = i_min # hx fuel outlet (fuel node 3)
        d_indicies[12] = i_min # hx fuel outlet (fuel node 4)

    # hx fuel inlet 
    if (t_current > tau_c_hx):

        # delay time
        t_d = t_current-tau_c_hx

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_current < diff_min):
                i_min = t[0]
                diff_min = t[1]-t_d
        
        d_indicies[7] = i_min # core fuel outlet (fuel node 1)
        d_indicies[8] = i_min # core fuel outlet (fuel node 2)
        d_indicies[9] = i_min # core fuel outlet (graphite node 1)
        d_indicies[10] = i_min # core fuel outlet (neutron density)

    # hx precursors 
    if (t_current > tau_l):

        # delay time
        t_d = t_current-tau_c_hx

        # find closest value
        i_min = 0
        diff_min = 9999.99999
        for t in enumerate(timeVec):
            if (abs(t[1]-t_current) < diff_min):
                i_min = t[0]
                diff_min = abs(t[1]-t_d)
        
        d_indicies[13] = i_min # precursor group 1
        d_indicies[14] = i_min # precursor group 2
        d_indicies[15] = i_min # precursor group 3
        d_indicies[16] = i_min # precursor group 4
        d_indicies[17] = i_min # precursor group 5
        d_indicies[18] = i_min # precursor group 6

    return d_indicies


def get_tIdx(t,tao,timeVec):
    td = t-tao
    diff_min = 999999.9999999
    idx = 0
    for t in enumerate(timeVec):
        diff = abs(td-t[1])
        if (diff<diff_min):
            diff_min = abs(td-t[1])
            idx = t[0]
    return idx



def main():

    # initial time

    # initial conditions
    y0 = [T0_s4, T0_rp, T0_rs, T0_f2, T0_p1,T0_p2, T0_p3, T0_p4, T0_t1, T0_t2,
          T0_rp, T0_s1, T0_s2, T0_s3, T0_s4, T0_p4, n_frac0, C0[0], 
          C0[1], C0[2], C0[3], C0[4], C0[5], T0_g1, T0_f1, T0_f2
    ]
     

    # initial delay terms 
    d0 = [T0_rp, T0_rs, T0_s4,  T0_s3, T0_s4, T0_t1, T0_t2, T0_f1, T0_f2,     \
          T0_g1, n_frac0, T0_p3, T0_p4, C0[0], C0[1], C0[2], C0[3], C0[4],    \
          C0[5]] 

    # solve
    backend = 'dopri5'
    r = ode(dydtMSRE).set_integrator(backend)

    sol_interim = []
    def solout(t, y):
        sol_interim.append([t, *y])
    r.set_solout(solout)

    t0 = 0.0
    t_start = t0
    sol = []
    y_next = []
    k = 2
    t_stop = 700.0
    d_new = [0]*19
    i = 0
    insert = 1.39e-4
    t_insert = 500.00
    while (t_start < t_stop):
        # take one step
        if (i == 0):
            r.set_initial_value(y0,t0).set_f_params(d0,0.0)
            r.integrate(0.1)
            sol.append(sol_interim[0])
            sol.append(sol_interim[1])
        else:
            t_start = sol[-1][0]
            if (t_start>=t_insert):
                r.set_initial_value(y_next,t_start).set_f_params(d_new,insert)
            else:
                r.set_initial_value(y_next,t_start).set_f_params(d_new,0.0)
            r.integrate(t_start+0.1)
            sol.append(sol_interim[1])

        # account for delays 
        # core fuel inlet
        idx_cf_in = 0
        if (t_start > tau_hx_c):
            idx_cf_in = get_tIdx(t_start,tau_hx_c,[s[0] for s in sol])
        sol[-1][16] = sol[idx_cf_in][8]

        # heat exchanger fuel inlet
        idx_hf_in = 0
        if (t_start > tau_c_hx):
            idx_hf_in = get_tIdx(t_start,tau_c_hx,[s[0] for s in sol])
        sol[-1][4] = sol[idx_hf_in][26]

        # heat exchanger coolant inlet
        idx_hc_in = 0
        if (t_start > tau_r_hx):
            idx_hc_in = get_tIdx(t_start,tau_r_hx,[s[0] for s in sol])
        sol[-1][11] = sol[idx_hc_in][2]

        # radiator coolant inlet
        idx_rc_in = 0
        if (t_start > tau_hx_r):
            idx_rc_in = get_tIdx(t_start,tau_hx_r,[s[0] for s in sol])
        sol[-1][1] = sol[idx_rc_in][15]

        # update delay terms 
        d_idx = get_delay_indicies(t_start,[s[0] for s in sol])

        # map solution to delay terms
        y_to_d_map = [1,2,14,13,14,8,9,24,25,23,16,6,7,17,18,19,20,21,22]
        for j in range(19):
            d_new[j] = sol[d_idx[j]][y_to_d_map[j]+1] 

        y_next = sol[-1][1:]
        sol_interim = []
        i += 1
        print(t_start)

    for i in range(len(sol)-6,len(sol)-1):
        i_rev = len(sol)-(10-i)
        print(f"t: {sol[i][0]}")
        print(f"n: {sol[i][17]}")
        print(f"T0_g1: {T0_g1}")
        print(f"T_cg: {sol[i][24]}")
        print(f"T_cf_in: {sol[i][16]}")
        print(f"T0_f1: {T0_f1}")
        print(f"T_cf1: {sol[i][25]}")
        print(f"T0_f2: {T0_f2}")
        print(f"T_cf2: {sol[i][26]}")
        T_cg = sol[i][24]
        T_cf1 = sol[i][25]
        T_cf2 = sol[i][26]
        print(f"rho: {(a_f/2)*((-T0_f1+T_cf1)+(-T0_f2+T_cf2)) + a_g*(-T0_g1+T_cg)}\n")

    print(sol[-5:])
    #rhos = [(a_f/2)*((-T0_f1+sol[i][25])+(-T0_f2+sol[i][26])) + a_g*(-T0_g1+sol[i][24]) for i in range(len(sol))]

    #for j in range(len(sol[0])):
    #    plt.plot([s[0] for s in sol],[s[j] for s in sol])
    #    plt.show()
    # 1: radiatior coolant inlet temp
    # 2: radiator coolant outlet temp
    # 3: radiator air outlet temp
    # 4: heat exchanger fuel inlet temp
    # 5: heat exchanger fuel node 1 temp
    # 6: heat exchanger fuel node 2 temp
    # 7: heat exchanger fuel node 3 temp
    # 8: heat exchanger fuel node 4 temp
    # 9: heat exchanger tube node 1 temp
    # 10: heat exchanger tube node 2 temp
    # 11: heat exchanger coolant inlet temp
    # 12: heat exchanger coolant node 1
    # 13: heat exchanger coolant node 2
    # 14: heat exchanger coolant node 3
    # 15: heat exchanger coolant node 4
    # 16: core fuel inlet temp
    # 17: k
    # 18: C1
    # 19: C2
    # 20: C3
    # 21: C4
    # 22: C5
    # 23: C6
    # 24: core graphite temp
    # 25: core fuel node 1 temp
    # 26: core fuel node 2 temp
    #of_interest = 17
    #plt.plot([s[0] for s in sol],[s[of_interest] for s in sol])
    #plt.show()

    for p in range(0,27):
        #generate id  
        fig,ax = plt.subplots()  #create a new figure
        tidx = [s[0] for s in enumerate(sol) if s[1][0] > 499.00]
        ax.plot([s[0] for s in sol if s[0] > 499.00],[s[p] for s in sol][tidx[0]:])
        fig.savefig(f"{p}.png")

    #for i in range(len(sol)):
    #    print(f"t: {sol[i][0]}, hx out: {sol[i][8]}, core in: {sol[i][16]}")

    return None 

def debug():

    # initial time
    t0 = 0.0

    # initial conditions
    y0 = [T0_s4, T0_rp, T0_rs, T0_f2, T0_p1,T0_p2, T0_p3, T0_p4, T0_t1, T0_t2,
          T0_rp, T0_s1, T0_s2, T0_s3, T0_s4, T0_p4, n_frac0, C0[0], 
          C0[1], C0[2], C0[3], C0[4], C0[5], T0_g1, T0_f1, T0_f2
    ]
    

    # initial delay terms 
    d0 = [T0_rp, T0_rs, T0_s4,  T0_s3, T0_s4, T0_t1, T0_t2, T0_f1, T0_f2,     \
          T0_g1, n_frac0, T0_p3, T0_p4, C0[0], C0[1], C0[2], C0[3], C0[4],    \
          C0[5]] 

    init = [i for i in y0]
    init.insert(0,0.0)
    dt = 0.0001
    sol = [init]
    derivs = []
    rhos = []
    #print(f"{y0}\n")
    for i in range(1):
        deriv = dydtMSRE(sol[-1][0],sol[-1][1:],d0)
        derivs.append(deriv)
        deriv.insert(0,sol[-1][0]+dt)
        new_sol = [sol[-1][j]+dt*deriv[j] for j in range(len(deriv))]
        new_sol[0] = sol[-1][0]+dt
        sol.append(new_sol)
    print(derivs)
    for i in range(len(derivs)):
        print(f"t: {derivs[i][0]}")    
        print(f"dTdt_cf_in: {derivs[i][16]}")
    #for i in range(10):
    #    print(f"t: {sol[i][0]}")
    #    print(f"n: {sol[i][17]}\n")
    #plt.plot([s[0] for s in sol[1:]],rhos)
    #plt.plot([s[0] for s in sol],[s[17] for s in sol])
    #plt.show()

    return None

def debug2():

    # initial time
    t0 = 0.0

    # initial conditions
    y0 = [T0_s4, T0_rp, T0_rs, T0_f2, T0_p1,T0_p2, T0_p3, T0_p4, T0_t1, T0_t2,
          T0_rp, T0_s1, T0_s2, T0_s3, T0_s4, T0_p4, n_frac0, C0[0], 
          C0[1], C0[2], C0[3], C0[4], C0[5], T0_g1, T0_f1, T0_f2
    ]
    

    # initial delay terms 
    d0 = [T0_rp, T0_rs, T0_s4,  T0_s3, T0_s4, T0_t1, T0_t2, T0_f1, T0_f2,     \
          T0_g1, n_frac0, T0_p3, T0_p4, C0[0], C0[1], C0[2], C0[3], C0[4],    \
          C0[5]] 

    dydt = dydtMSRE(t0,y0,d0)
    for i in range(len(dydt)):
        print(f"{y0[i]},{dydt[i]}")

    return None

#debug()
#debug2()
main()