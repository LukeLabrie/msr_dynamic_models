from parameters import *
import numpy as np
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt

def dydtMSRE(t, y, T_out_rc_delay, T_out_air_delay, T_in_rc_delay, T_hc3_delay, T_hc4_delay, \
    T_ht1_delay, T_ht2_delay, T_cf1_delay, T_cf2_delay, T_cg_delay, n_delay,  \
    T_hf3_delay, T_hf4_delay, C1_delay, C2_delay, C3_delay, C4_delay,         \
    C5_delay, C6_delay):

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
    T_ht2, T_in_hc, T_hc1, T_hc2, T_hc3, T_hc4, T_in_cf, rho , n, C1, C2,    \
    C3, C4, C5, C6, T_cg, T_cf1, T_cf2 = y
    
    # derivatives 

    # need to handle time delays, perhaps take one-step at a time, update y_0
    # or use function as input which calculates necessary delay terms 

    dydt = [ 
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc4_delay + (hA_sn/mcp_sn)*T_ht1_delay + (W_s/mn_s)*T_hc3_delay,                                  # T_in_rc
    -((W_rp/mn_rp)+(hA_rpn/mcp_rpn))*T_out_rc + (hA_rpn/mcp_rpn)*T_out_air + (W_rp/mn_rp)*T_in_rc,                                   # T_out_rc
    -((W_rs/mn_rs)+(hA_rsn/mcp_rsn))*T_out_air + (hA_rsn/mcp_rsn)*T_out_rc + (W_rs/mn_rs)*Trs_in,                                    # T_out_air
    -((W_f/mn_f)+(hA_fg*k_2/mcp_f2))*T_cf2_delay + (hA_fg*k_2/mcp_f2)*T_cg_delay + (W_f/mn_f)*T_cf1_delay + (k_f2*P/mcp_f2)*n_delay, # T_in_hf
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf1 + (hA_pn)*T_ht1 + (W_p/mn_p)*T_in_hf,                                                         # T_hf1
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf2 + (hA_pn)*T_ht1 + (W_p/mn_p)*T_hf1,                                                           # T_hf2
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf3 + (hA_pn)*T_ht2 + (W_p/mn_p)*T_hf2,                                                           # T_hf3
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf4 + (hA_pn)*T_ht2 + (W_p/mn_p)*T_hf3,                                                           # T_hf4
    -2*((hA_pn/mcp_tn)+(hA_sn/mcp_tn))*T_ht1 + (2*hA_pn/mcp_tn)*T_hf1 + (2*hA_sn/mcp_tn)*T_hc3,                                      # T_ht1
    -2*((hA_pn/mcp_tn)+(hA_sn/mcp_tn))*T_ht2 + (2*hA_pn/mcp_tn)*T_hf3 + (2*hA_sn/mcp_tn)*T_hc1,                                      # T_ht2
    -((W_rp/mn_rp)+(hA_rpn/mcp_rpn))*T_out_rc_delay + (hA_rpn/mcp_rpn)*T_out_air_delay + (W_rp/mn_rp)*T_in_rc_delay,                 # T_in_hc
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc1 + (hA_sn/mcp_sn)*T_ht2 + (W_s/mn_s)*T_in_hc,                                                  # T_hc1
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc2 + (hA_sn/mcp_sn)*T_ht2 + (W_s/mn_s)*T_hc1,                                                    # T_hc2
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc3 + (hA_sn/mcp_sn)*T_ht1 + (W_s/mn_s)*T_hc2,                                                    # T_hc3
    -((W_s/mn_s)+(hA_sn/mcp_sn))*T_hc4 + (hA_sn/mcp_sn)*T_ht1 + (W_s/mn_s)*T_hc3,                                                    # T_hc4
    -((W_p/mn_p)+(hA_pn/mcp_pn))*T_hf4_delay + (hA_pn)*T_ht2_delay + (W_p/mn_p)*T_hf3_delay,                                         # T_in_cf
    -((a_f/2)*(((W_f/mn_f)+(hA_fg*k_1/mcp_f1))*T_cf1 + (hA_fg*k_1/mcp_f1)*T_cg + (k_f1*P*n/mcp_f1)-((W_f/mn_f) + \
    (hA_fg*k_2/mcp_f2))*T_cf2 + (hA_fg*k_2/mcp_f2)*T_cg + (k_f2*P*n/mcp_f2))+a_g*((hA_fg/mcp_g1)*(T_cf1 - T_cg) + k_g*P*n/mcp_g1)),  # rho  
    (rho-beta_t)*n/Lam+lam[0]*C1+lam[1]*C2+lam[2]*C3+lam[3]*C4+lam[4]*C5+lam[5]*C6,                                                  # n (no source insertion)
    (beta[0]/Lam)*n-(lam[0]-1/tau_c)*C1+C1_delay*np.exp(-lam[0]*tau_l)/tau_c,                                                           # C1
    (beta[1]/Lam)*n-(lam[1]-1/tau_c)*C2+C2_delay*np.exp(-lam[1]*tau_l)/tau_c,                                                           # C2
    (beta[2]/Lam)*n-(lam[2]-1/tau_c)*C3+C3_delay*np.exp(-lam[2]*tau_l)/tau_c,                                                           # C3
    (beta[3]/Lam)*n-(lam[3]-1/tau_c)*C4+C4_delay*np.exp(-lam[3]*tau_l)/tau_c,                                                           # C4
    (beta[4]/Lam)*n-(lam[4]-1/tau_c)*C5+C5_delay*np.exp(-lam[4]*tau_l)/tau_c,                                                           # C5
    (beta[5]/Lam)*n-(lam[5]-1/tau_c)*C6+C6_delay*np.exp(-lam[5]*tau_l)/tau_c,                                                           # C6
    (hA_fg/mcp_g1)*(T_cf1 - T_cg) + k_g*P*n/mcp_g1,                                                                                  # T_cg
    -((W_f/mn_f)+(hA_fg*k_1/mcp_f1))*T_cf1 + (hA_fg*k_1/mcp_f1)*T_cg + (k_f1*P*n/mcp_f1),                                            # T_cf1   
    -((W_f/mn_f)+(hA_fg*k_2/mcp_f2))*T_cf2 + (hA_fg*k_2/mcp_f2)*T_cg + (k_f2*P*n/mcp_f2)                                             # T_cf2   
    ]

    return dydt

def get_delay_indicies(t,timeVec):

    # indicies of delay
    d_indicies = [0]*19

    # radiator coolant outlet 
    if (t > tau_r_hx):

        # delay time
        t_d = t-tau_r_hx

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_d < diff_min):
                i_min = t[0]
        
        d_indicies[0] = i_min # radiator coolant outlet
        d_indicies[1] = i_min # radiator air outlet 
        d_indicies[2] = i_min # radiator coolant inlet 

    # radiator coolant inlet 
    if (t > tau_hx_r):

        # delay time
        t_d = t-tau_hx_r

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_d < diff_min):
                i_min = t[0]
        
        d_indicies[3] = i_min # hx coolant outlet (coolant node 3)
        d_indicies[4] = i_min # hx coolant outlet (coolant node 4)
        d_indicies[5] = i_min # hx coolant outlet (tube node 1)

    # core fuel inlet 
    if (t > tau_hx_c):

        # delay time
        t_d = t-tau_hx_c

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_d < diff_min):
                i_min = t[0]
        
        d_indicies[6] = i_min # hx fuel outlet (tube node 2)
        d_indicies[11] = i_min # hx fuel outlet (fuel node 3)
        d_indicies[12] = i_min # hx fuel outlet (fuel node 4)

    # hx fuel inlet 
    if (t > tau_c_hx):

        # delay time
        t_d = t-tau_c_hx

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_d < diff_min):
                i_min = t[0]
        
        d_indicies[7] = i_min # core fuel outlet (fuel node 1)
        d_indicies[8] = i_min # core fuel outlet (fuel node 2)
        d_indicies[9] = i_min # core fuel outlet (graphite node 1)
        d_indicies[10] = i_min # core fuel outlet (neutron density)

    # hx precursors 
    if (t > tau_l):

        # delay time
        t_d = t-tau_c_hx

        # find closest value
        i_min = 0
        diff_min = math.inf
        for t in enumerate(timeVec):
            if (t[1]-t_d < diff_min):
                i_min = t[0]
        
        d_indicies[13] = i_min # precursor group 1
        d_indicies[14] = i_min # precursor group 2
        d_indicies[15] = i_min # precursor group 3
        d_indicies[16] = i_min # precursor group 4
        d_indicies[17] = i_min # precursor group 5
        d_indicies[18] = i_min # precursor group 6

    return d_indicies



def main():

    # initial time
    t0 = 0.0

    # initial conditions
    y0 = [T0_s4, T0_rp, T0_rs, T0_f2, T0_p1,T0_p2, T0_p3, T0_p4, T0_t1, T0_t2,
          T0_rp, T0_s1, T0_s2, T0_s3, T0_s4, T0_p4, rho_0, n_frac0, C0[0], 
          C0[1], C0[2], C0[3], C0[4], C0[5], T0_g1, T0_f1, T0_f2
    ]
     

    # initial delay terms 
    d0 = [T0_rp, T0_rs, T0_s4,  T0_s3, T0_s4, T0_t1, T0_t2, T0_f1, T0_f2,     \
          T0_g1, n_frac0, T0_p3, T0_p4, C0[0], C0[1], C0[2], C0[3], C0[4],    \
          C0[5]] 

    # solve
    sol = solve_ivp(dydtMSRE,[0,10],y0,args=d0,dense_output=True)    
    plt.plot(sol.t,sol.y[3])
    plt.show()
    print(sol.t)

    # not working 
    
    #backend = 'dopri5'
    #r = ode(dydtMSRE).set_integrator(backend)

    #sol_interim = []
    #def solout(t, y):
    #    sol_interim.append([t, *y])
    #r.set_solout(solout)

    #r.set_initial_value(y0,t0).set_f_params(d0)
    #r.integrate(10.00)

    #plt.plot([s[0] for s in sol_interim],[s[3] for s in sol_interim])
    #plt.show()
    #for i in range(0,3):
    #    print(f"{sol_interim[i]}\n")

    return None 

main()