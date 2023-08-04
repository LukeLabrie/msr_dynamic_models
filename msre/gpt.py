import numpy as np
import pandas as pd
import math

pi = math.pi

# Perturbations
# SOURCE INSERTION
# No source insertion
sourcedata = np.array([0, 0, 0])
sourcetime = np.array([0, 50, 100])
# % 1 (n/no)/s for 10 seconds
# sourcedata = np.array([0, 10, 0])
# sourcetime = np.array([0, 10, 20])
source = pd.Series(sourcedata, index=sourcetime)

# REACTIVITY INSERTION
# No reactivity insertion
simtime = 10
reactdata = np.array([0, 5E-4])
reacttime = np.array([0, 2500])
# Periodic 60 PCM for 50 seconds
# simtime = 500
# periodic = np.array([[0, 0], [50, 6e-4], [100, 0], [150, -6e-4], [200, 0], [250, 6e-4], [300, 0], [350, -6e-4], [400, 0]])
# reactdata = periodic[:, 1]
# reacttime = periodic[:, 0]
# Step up 60 pcm
# simtime = 1000
# reactdata = np.array([0, 6e-3])
# reacttime = np.array([0, 300])
# # Step down -60 pcm for 10 sec
# simtime = 100
# reactdata = np.array([0, -6e-4])
# reacttime = np.array([0, 50])
# # Pulse 600 pcm for 0.1 sec
# simtime = 30
# reactdata = np.array([0, 6e-3, 0])
# reacttime = np.array([0, 10, 10.1])

react = pd.Series(reactdata, index=reacttime)

ts_max = 1e-1  # maximum timestep (s)

# NEUTRONICS DATA
tau_l = 16.73  # ORNL-TM-0728 %16.44; % (s)
tau_c = 8.46  # ORNL-TM-0728 %8.460; % (s)
P = 8  # Thermal Power in MW ORNL-TM-1070, p.2
n_frac0 = 1  # initial fractional neutron density n/n0 (n/cm^3/s)
Lam = 2.400E-04  # mean generation time ORNL-TM-1070 p.15 U235
# Lam = 4.0E-04;  # mean generation time ORNL-TM-1070 p.15 U233
lam = np.array([1.240E-02, 3.05E-02, 1.11E-01, 3.01E-01, 1.140E+00, 3.014E+00])
beta = np.array([0.000223, 0.001457, 0.001307, 0.002628, 0.000766, 0.00023])  # U235
# beta = np.array([0.00023, 0.00079, 0.00067, 0.00073, 0.00013, 0.00009])  # U233
beta_t = np.sum(beta)  # total delayed neutron fraction MSRE
rho_0 = beta_t - np.dot(beta, np.exp(-lam * tau_l) - 1.0) / (lam * tau_c)  # reactivity change in going from stationary to circulating fuel
C0 = beta / Lam * (1.0 / (lam - (np.exp(-lam * tau_l) - 1.0) / tau_c))

# Feedback co-efficients
a_f = -8.71E-05  # U235 (drho/°C) fuel salt temperature-reactivity feedback coefficient ORNL-TM-1647 p.3 % -5.904E-05; % ORNL-TM-0728 p. 101 %
a_g = -6.66E-05  # U235 (drho/°C) graphite temperature-reactivity feedback coefficient ORNL-TM-1647 p.3 % -6.624E-05; % ORNL-TM-0728 p.101

# CORE HEAT TRANSFER PARAMETERS
# FUEL PARAMETERS - DONE
vdot_f = 7.5708E-02  # ORNL-TM-0728 % 7.571e-2; % vol. flow rate (m^3/s) ORNL-TM-1647 p.3, ORNL-TM-0728 p.12
rho_f = 2.14647E+03  # (partially enriched U-235)ORNL-TM-0728 p.8 2.243E+03; % (Th-U) density of fuel salt (kg/m^3) ORNL-TM-0728 p.8
W_f = 1.623879934566580e+02  # 1.83085e+02;%vdot_f*rho_f; % 182.78; % calcd from m_dot*cp*delT=P; vdot_f*rho_f; % fuel flow rate (kg/s)
# tau_f_c  = tau_c; % ORNL-TM-0728 % 8.45; % transit time of fuel in core (s) ORNL-TM-1070 p.15, TDAMSRE p.5
m_f = W_f * tau_c  # fuel mass in core (kg)
nn_f = 2  # number of fuel nodes in core model
mn_f = m_f / nn_f  # fuel mass per node (kg)
# cp_f     = 4.2*9/5; % (MJ/deg-C) total fuel heat capacity TDAMSRE p.5
scp_f = 1.9665E-3  # specific heat capacity of fuel salt (MJ/kg-C) ORNL-TM-0728 p.8

# Core Upflow - DONE
v_g = 1.95386  # graphite volume(m^3) ORNL-TM-0728 p. 101
rho_g = 1.860E3  # graphite density (kg/m^3) ORNL-3812 p.77, ORNL-TM-0728 p.87
m_g = v_g * rho_g  # graphite mass (kg)
cp_g = 3.6 * 9 / 5  # TDAMSRE p.5 graphite total heat capacity (MW-s/C) ORNL-TM-1647 p.3
scp_g = 1.773E-3  # cp_g/m_g; % graphite specific heat capacity (MW-s/kg-C) ORNL-TM-1647 p.3
mcp_g1 = m_g * scp_g  # (mass of material x heat capacity of material) of graphite per lump (MW-s/°C)
mcp_f1 = mn_f * scp_f  # (mass of material x heat capacity of material) of fuel salt per lump (MW-s/°C)
mcp_f2 = mn_f * scp_f  # (mass of material x heat capacity of material) of fuel salt per lump (MW-s/°C)
hA_fg = 0.02 * 9 / 5  # (fuel to graphite heat transfer coeff x heat transfer area) (MW/°C) ORNL-TM-1647 p.3, TDAMSRE p.5
k_g = 0.07  # fraction of total power generated in the graphite  ORNL-TM-0728 p.9
k_1 = 0.5  # fraction of heat transferred from graphite which goes to the first fuel lump
k_2 = 0.5  # fraction of heat transferred from graphite which goes to the second fuel lump
k_f = 0.93  # fraction of heat generated in fuel - that generated in the external loop ORNL-TM-0728 p.9
k_f1 = k_f / nn_f  # fraction of total power generated in lump f1
k_f2 = k_f / nn_f  # fraction of total power generated in lump f2

# Heat Exchanger - DONE
# Geometry
d_he = 16  # (in) he diameter ORNL-TM-0728 p. 164
h_he = 72  # (in) active height % 96; %(in) he height ORNL-TM-0728 p. 164
od_tube = 0.5  # (in) coolant tube OD ORNL-TM-0728 p. 164
id_tube = od_tube - 2 * 0.042  # (in) coolant tube ID ORNL-TM-0728 p. 164
n_tube = 159  # number of coolant tubes ORNL-TM-0728 p. 164
a_tube = 254 * 144  # (in^2) total area of tubes ORNL-TM-0728 p. 164
l_tube = a_tube / n_tube / (np.pi * od_tube)  # (in) tube length
v_tube = n_tube * np.pi * (od_tube / 2) ** 2 * l_tube  # (in^3) hx shell volume occupied by tubes
v_cool = n_tube * np.pi * (id_tube / 2) ** 2 * l_tube  # (in^3) hx volume occupied by coolant
v_he = (d_he / 2) ** 2 * np.pi * h_he  # (in^3) volume of heat exchanger shell
v_he_fuel = v_he - v_tube  # (in^3) volume available to fuel in shell

# Unit conversions
in_m = 1.63871e-5  # 1 cubic inch = 1.63871e-5 cubic meters

# PRIMARY FLOW PARAMETERS - DONE
W_p = W_f  # fuel flow rate (kg/s)

m_p = v_he_fuel * in_m * rho_f  # fuel mass in PHE (kg)
nn_p = 4  # number of fuel nodes in PHE
mn_p = m_p / nn_p  # fuel mass per node (kg)
cp_p = scp_f  # fuel heat capacity (MJ/(kg-C))

# SECONDARY FLOW PARAMETERS - DONE
vdot_s = 5.36265E-02  # ORNL-TM-0728 p. 164 % 5.236E-02; % coolant volume flow rate (m^3/s) ORNL-TM-1647 p.3
rho_s = 1.922e3  # coolant salt density (kg/m^3) ORNL-TM-0728 p.8
W_s = 1.005793369810108e+02  # vdot_s*rho_s; % calcd from mdot*cp*delT; vdot_s*rho_s; % coolant flow rate (kg/s) ORNL-TM-1647 p.3

m_s = v_cool * in_m * rho_s  # coolant mass in PHE (kg)
nn_s = 4  # number of coolant nodes in PHE
mn_s = m_s / nn_s  # coolant mass per node (kg)
scp_s = 2.39E-3  # cp_s/m_s; % specific heat capacity of coolant (MJ/(kg-C) ORNL-TM-0728 p.8

A_phe = 2.359E+01  # effective area for heat transfer (primary and secondary, m^2) ORNL-TM-0728 p.164

ha_p = 6.480E-01  # heat transfer*area coefficient from primary to tubes (MW/C) ORNL-TM-1647 p.3
ha_s = 3.060E-01  # heat transfer*area coefficient from tubes to secondary (MW/C) ORNL-TM-1647 p.3

# Primary Side
mcp_pn = mn_p * cp_p  # (mass of material x heat capacity of material) of fuel salt per lump in MW-s/°C
hA_pn = ha_p / nn_s  # 3.030; % (primary to tube heat transfer coeff x heat transfer area) in MW/°C

# Tubes - DONE
nn_t = 2  # number of nodes of tubes in the model
rho_tube = 8.7745E+03  # (kg/m^3) density of INOR-8 ORNL-TM-0728 p.20
m_tn = (v_tube - v_cool) * in_m * rho_tube / nn_t  # mass of tubes (kg)
scp_t = 5.778E-04  # specific heat capacity of tubes (MJ/(kg-C)) ORNL-TM-0728 p.20
mcp_tn = m_tn * scp_t  # mass*(heat capacity) of tubes per lump in MW-s/°C

# Secondary Side - DONE
mcp_sn = mn_s * scp_s  # (mass of material x heat capacity of material) of coolant salt per lump in MW-s/°C
hA_sn = ha_s / nn_s  # (tube to secondary heat transfer coeff x heat transfer area) in MW/°C

# Initial conditions - DONE
# Primary nodes
Tp_in = 6.3222E+02  # in °C ORNL-TM-1647 p.2
T0_p4 = 6.5727E+02  # 6.5444E+02; % in °C 6.461904761904777e+02; ORNL-TM-1647 p.2
T0_p1 = Tp_in + (T0_p4 - Tp_in) / 4  # in °C
T0_p2 = Tp_in + 2 * (T0_p4 - Tp_in) / 4  # in °C
T0_p3 = Tp_in + 3 * (T0_p4 - Tp_in) / 4  # in °C

# Secondary nodes
Ts_in = 5.4611E+02  # in °C ORNL-TM-1647 p.2
T0_s4 = 5.7939E+02  # in °C ORNL-TM-1647 p.2
T0_s1 = Ts_in + (T0_s4 - Ts_in) / nn_s  # in °C
T0_s2 = Ts_in + 2 * (T0_s4 - Ts_in) / nn_s  # in °C
T0_s3 = Ts_in + 3 * (T0_s4 - Ts_in) / nn_s  # in °C
# Tube nodes
T0_t1 = (T0_p1 * hA_pn + T0_s3 * hA_sn) / (hA_pn + hA_sn)  # in °C
T0_t2 = (T0_p3 * hA_pn + T0_s1 * hA_sn) / (hA_pn + hA_sn)  # in °C

# Radiator Parameters - DONE

# Initial conditions - DONE
# Primary nodes
Trp_in = T0_s4  # 5.933E+02; % in °C ORNL-TM-1647 p.2
T0_rp = Ts_in  # in °C ORNL-TM-1647 p.2

# Secondary nodes - DONE
Trs_in = 37.78  # (C) air inlet temperature ORNL-TM-1647 p.2
T0_rs = 148.9  # (C) air exit temperature ORNL-TM-1647 p.2

# Radiator Geometry
od_rad = 0.01905  # (m) outer diameter of tubes in the radiator ORNL-TM-0728 p.296
tube_wall_thick = 0.0018288  # (m) thickness of tubes in the radiator ORNL-TM-0728 p.296
id_rad = od_rad - 2 * tube_wall_thick
n_rtubes = 120  # number of tubes in the radiator (rows times tubes per row) ORNL-TM-0728 p.296
l_rtube = 9.144  # (m) length of tubes in the radiator ORNL-TM-0728 p.296
v_rp = pi * (id_rad / 2) ** 2 * l_rtube * n_rtubes  # volume available to salt in the radiator
# v_rtube = pi * (od_rad / 2) ** 2 * l_rtube * n_rtubes - v_rp  # volume of metal in radiator tubes *TUBES NOT MODELED

n_tpr = 12  # number of tubes per row in the radiator matrix
n_row = 10  # number rows in the radiator matrix
tube_space = 0.0381  # (m) spacing between tubes and rows of matrix
v_rs = (n_row * od_rad + (n_row - 1) * tube_space) * (n_tpr * od_rad + (n_tpr - 1) * tube_space) * l_rtube  # volume of air inside radiator

# PRIMARY FLOW PARAMETERS - DONE
W_rp = W_s  # coolant salt flow rate (kg/s)
m_rp = v_rp * rho_s  # coolant salt mass in rad (kg)
nn_rp = 1  # number of coolant salt nodes in the radiator
mn_rp = m_rp / nn_rp  # coolant mass per node (kg)
cp_rp = scp_s  # coolant specific heat capacity (MJ/(kg-C))

# SECONDARY FLOW PARAMETERS - DONE
vdot_rs = 94.389  # ORNL-TM-0728 p. 296; 78.82; % air volume flow rate (m^3/s) ORNL-TM-1647 p.2
rho_rs = 1.1237  # air density (kg/m^3) REFPROP (310K and 0.1MPa)
W_rs = vdot_rs * rho_rs  # air flow rate (kg/s)

m_rs = v_rs * rho_rs  # coolant air mass in rad (kg)
nn_rs = 1  # number of coolant nodes in rad
mn_rs = m_rs / nn_rs  # coolant mass per node (kg)
scp_rs = 1.0085E-3  # (MJ/kg-C) specific heat capacity of air at (air_out+air_in)/2 REFPROP

A_rad = 6.503E1  # (m^2) surface area of the radiator ORNL-TM-0728 p.14
h_roverall = P / A_rad / ((T0_rp + Trp_in) / 2 - (T0_rs + Trs_in) / 2)  # cald as: P/A_rad/((T0_rp+Trp_in)/2-(T0_rs+Trs_in)/2)  3.168E-4; % (MW/m^2-C) polimi thesis

# Primary Side
mcp_rpn = mn_rp * cp_rp  # (mass of material x heat capacity of material) of fuel salt per lump in MW-s/°C
hA_rpn = h_roverall * A_rad / nn_rs  # 3.030; % (primary to secondary heat transfer coeff x heat transfer area) in MW/°C

# Secondary Side - DONE
mcp_rsn = mn_rs * scp_rs  # (mass of material x heat capacity of material) of coolant salt per lump in MW-s/°C
hA_rsn = h_roverall * A_rad / nn_rs  # (tube to secondary heat transfer coeff x heat transfer area) in MW/°C

# Pure time delays between components - DONE
tau_hx_c = 8.67  # (sec) delay from hx to core TDAMSRE p.6
tau_c_hx = 3.77  # (sec) subtracted 1 sec for external loop power generation node resident time; delay from core to fuel hx TDAMSRE p.6
tau_hx_r = 4.71  # (sec) fertile hx to core TDAMSRE p.6
tau_r_hx = 8.24  # (sec) core to fertile hx TDAMSRE p.6

first_val = (rho_0 - beta_t) * n_frac0 / Lam + lam[0] * C0[0] + lam[1] * C0[1] + lam[2] * C0[2] + lam[3] * C0[3] + lam[4] * C0[4] + lam[5] * C0[5]
n_frac_ss = ((rho_0 - beta_t) * n_frac0 / Lam + first_val * (1 - np.exp(-simtime / tau_c)) / (1 - np.exp(-simtime / Lam / tau_c))) / (1 + (first_val / Lam / tau_c) * (1 - np.exp(-simtime / tau_c)))  # TDAMSRE p.6
n_frac1_ss = C0[0] * (1 - np.exp(-simtime / tau_c)) / (1 + (C0[0] / tau_c) * (1 - np.exp(-simtime / tau_c)))
n_frac2_ss = C0[1] * (1 - np.exp(-simtime / tau_c)) / (1 + (C0[1] / tau_c) * (1 - np.exp(-simtime / tau_c)))
n_frac3_ss = C0[2] * (1 - np.exp(-simtime / tau_c)) / (1 + (C0[2] / tau_c) * (1 - np.exp(-simtime / tau_c)))
n_frac4_ss = C0[3] * (1 - np.exp(-simtime / tau_c)) / (1 + (C0[3] / tau_c) * (1 - np.exp(-simtime / tau_c)))
n_frac5_ss = C0[4] * (1 - np.exp(-simtime / tau_c)) / (1 + (C0[4] / tau_c) * (1 - np.exp(-simtime / tau_c)))
n_frac6_ss = C0[5] * (1 - np.exp(-simtime / tau_c)) / (1 + (C0[5] / tau_c) * (1 - np.exp(-simtime / tau_c)))
# Radiator - DONE

# Time step
t = np.arange(0, simtime, ts_max)

# Initialize neutron fraction array with a scalar value
n_frac_n = 1.0  # Initial neutron fraction (n/n0)

nr = len(t)

# Create arrays to store the results
Power_n = np.zeros(len(t))
T_p1n = np.zeros(len(t))
T_p2n = np.zeros(len(t))
T_p3n = np.zeros(len(t))
T_p4n = np.zeros(len(t))
T_s1n = np.zeros(len(t))
T_s2n = np.zeros(len(t))
T_s3n = np.zeros(len(t))
T_s4n = np.zeros(len(t))
T_rp_n = np.zeros(len(t))
T_r4n = np.zeros(len(t))
n_frac_n = np.zeros(len(t))
feedback_rho = np.zeros(len(t))

# Assign initial values
T_p1n[0] = T0_p1
T_p2n[0] = T0_p2
T_p3n[0] = T0_p3
T_p4n[0] = T0_p4
T_s1n[0] = T0_s1
T_s2n[0] = T0_s2
T_s3n[0] = T0_s3
T_s4n[0] = T0_s4
T_rp_n[0] = Trp_in
T_r4n[0] = T0_rp
n_frac_n[0] = n_frac0
feedback_rho[0] = 0.0

n_frac_n = np.ones(nr) * n_frac0
n_frac1_n = np.ones(nr) * 0
n_frac2_n = np.ones(nr) * 0
n_frac3_n = np.ones(nr) * 0
n_frac4_n = np.ones(nr) * 0
n_frac5_n = np.ones(nr) * 0
n_frac6_n = np.ones(nr) * 0

# Initializing power variables
Power_n = np.ones(nr) * P
Qn_n = np.ones(nr) * P * (1 - k_g)
Qp_n = np.ones(nr) * P * k_g * k_1
Qp2_n = np.ones(nr) * P * k_g * k_2

# Initializing heat exchanger variables
Q_dot_hx_n = np.ones(nr) * 0
Q_dot_hx4_n = np.ones(nr) * 0

# Initializing radiator variables
Q_dot_rn_n = np.ones(nr) * 0
Q_dot_rsn_n = np.ones(nr) * 0

# Simulation loop
for i in range(1, nr):
    ts = t[i] - t[i - 1]
    rho_ss = rho_s - beta_t * n_frac_n[i - 1]

    n_frac_n[i] = ((rho_0 - beta_t) * n_frac_n[i - 1] / Lam + rho_ss * ts / (1 + ts / Lam)) / (1 + rho_ss * ts / (1 + ts / Lam))

    # Reactivity insertions
    if react.index[0] <= t[i] < react.index[-1]:
        idx = np.where(t[i] >= react.index)[0][-1]
        rho_0 += react.values[idx]
        rho_ss = (rho_0 - beta_t) * n_frac_n[i - 1] / Lam + (C0[0] * np.exp(-t[i] / tau_c) + lam[0] * C0[0] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[1] * np.exp(-t[i] / tau_c) + lam[1] * C0[1] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[2] * np.exp(-t[i] / tau_c) + lam[2] * C0[2] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[3] * np.exp(-t[i] / tau_c) + lam[3] * C0[3] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[4] * np.exp(-t[i] / tau_c) + lam[4] * C0[4] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[5] * np.exp(-t[i] / tau_c) + lam[5] * C0[5] / tau_c * (1 - np.exp(-t[i] / tau_c)))
        n_frac_n[i] = ((rho_0 - beta_t) * n_frac_n[i - 1] / Lam + rho_ss * ts / (1 + ts / Lam)) / (1 + rho_ss * ts / (1 + ts / Lam))
        n_frac1_n[i] = (C0[0] * ts / (1 + ts / tau_c)) / (1 + C0[0] * ts / (1 + ts / tau_c))
        n_frac2_n[i] = (C0[1] * ts / (1 + ts / tau_c)) / (1 + C0[1] * ts / (1 + ts / tau_c))
        n_frac3_n[i] = (C0[2] * ts / (1 + ts / tau_c)) / (1 + C0[2] * ts / (1 + ts / tau_c))
        n_frac4_n[i] = (C0[3] * ts / (1 + ts / tau_c)) / (1 + C0[3] * ts / (1 + ts / tau_c))
        n_frac5_n[i] = (C0[4] * ts / (1 + ts / tau_c)) / (1 + C0[4] * ts / (1 + ts / tau_c))
        n_frac6_n[i] = (C0[5] * ts / (1 + ts / tau_c)) / (1 + C0[5] * ts / (1 + ts / tau_c))

    # Reactivity feedback
    alpha_Tp = alpha_T * (1 + 2 * beta_T * (T_p4n[i - 1] - T_0))
    rho_0 -= alpha_Tp * (T_p4n[i - 1] - T_0)

    # Feedback for reactivity - Core
    # reactivity beta_eff(ieff) % - partial derivative of beta with respect to a feedback - ref. TDAMSRE p. 4
    ieff = n_frac_n[i] - n_frac_ss

    # power feedback reactivity
    if reactivity == 'linear':
        rho_0 -= beta_eff(ieff) * (Power_n[i - 1] - P)
    elif reactivity == 'quadratic':
        rho_0 -= beta_eff(ieff) * (Power_n[i - 1] - P) + gamma_eff(ieff) * (Power_n[i - 1] - P) ** 2

    # Temperature feedback reactivity
    rho_0 -= beta_t * (T_p4n[i - 1] - T_0)

    # Set temperature and reactivity
    T_0 = T_p4n[i - 1]
    T_s0 = T_s4n[i - 1]
    rho_ss = (rho_0 - beta_t) * n_frac_n[i - 1] / Lam + (C0[0] * np.exp(-t[i] / tau_c) + lam[0] * C0[0] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[1] * np.exp(-t[i] / tau_c) + lam[1] * C0[1] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[2] * np.exp(-t[i] / tau_c) + lam[2] * C0[2] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[3] * np.exp(-t[i] / tau_c) + lam[3] * C0[3] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[4] * np.exp(-t[i] / tau_c) + lam[4] * C0[4] / tau_c * (1 - np.exp(-t[i] / tau_c))) + (C0[5] * np.exp(-t[i] / tau_c) + lam[5] * C0[5] / tau_c * (1 - np.exp(-t[i] / tau_c)))

    # Power calculations
    Qn_n[i] = Power_n[i - 1] * (1 - k_g)
    Qp_n[i] = Power_n[i - 1] * k_g * k_1
    Qp2_n[i] = Power_n[i - 1] * k_g * k_2

    # Heat exchanger
    # Calculations - Primary flow
    Q_dot_hx_n[i] = ha_p * (T_p4n[i - 1] - T_s1n[i - 1])  # MW - TDAMSRE p.7
    # mass flow - primary flow
    mcp_pn[i] = mn_p * cp_p  # (mass of material x heat capacity of material) of fuel salt per lump in MW-s/°C
    # new temperatures - primary flow
    T_p1n[i] = T_p1n[i - 1] + ts / tau_hx_c * (T_0 - T_p1n[i - 1]) + ts / mcp_pn[i] * Q_dot_hx_n[i - 1]
    T_p2n[i] = T_p2n[i - 1] + ts / tau_hx_c * (T_p1n[i - 1] - T_p2n[i - 1]) + ts / mcp_pn[i] * Q_dot_hx_n[i - 1]
    T_p3n[i] = T_p3n[i - 1] + ts / tau_hx_c * (T_p2n[i - 1] - T_p3n[i - 1]) + ts / mcp_pn[i] * Q_dot_hx_n[i - 1]
    T_p4n[i] = T_p4n[i - 1] + ts / tau_hx_c * (T_p3n[i - 1] - T_p4n[i - 1]) + ts / mcp_pn[i] * Q_dot_hx_n[i - 1]
    # Calculations - Secondary flow
    Q_dot_hx4_n[i] = ha_s * (T_p3n[i - 1] - T_s1n[i - 1])  # MW - TDAMSRE p.7
    # mass flow - secondary flow
    mcp_sn[i] = mn_s * scp_s  # (mass of material x heat capacity of material) of coolant salt per lump in MW-s/°C
    # new temperatures - secondary flow
    T_s1n[i] = T_s1n[i - 1] + ts / tau_hx_c * (T_s4n[i - 1] - T_s1n[i - 1]) + ts / mcp_sn[i] * Q_dot_hx4_n[i - 1]
    T_s2n[i] = T_s2n[i - 1] + ts / tau_hx_c * (T_s1n[i - 1] - T_s2n[i - 1]) + ts / mcp_sn[i] * Q_dot_hx4_n[i - 1]
    T_s3n[i] = T_s3n[i - 1] + ts / tau_hx_c * (T_s2n[i - 1] - T_s3n[i - 1]) + ts / mcp_sn[i] * Q_dot_hx4_n[i - 1]
    T_s4n[i] = T_s4n[i - 1] + ts / tau_hx_c * (T_s3n[i - 1] - T_s4n[i - 1]) + ts / mcp_sn[i] * Q_dot_hx4_n[i - 1]
    T_t1n[i] = (T_p1n[i] * hA_pn + T_s3n[i] * hA_sn) / (hA_pn + hA_sn)
    T_t2n[i] = (T_p3n[i] * hA_pn + T_s1n[i] * hA_sn) / (hA_pn + hA_sn)

    # Radiator
    # Calculations
    Q_dot_rn_n[i] = hA_rpn * (T_rp_n[i - 1] - T_rs_n[i - 1])  # MW - TDAMSRE p.7
    # mass flow
    mcp_rpn[i] = mn_rp * cp_rp  # (mass of material x heat capacity of material) of coolant salt per lump in MW-s/°C
    # new temperatures
    T_rp_n[i] = T_rp_n[i - 1] + ts / tau_hx_r * (T_r4n[i - 1] - T_rp_n[i - 1]) + ts / mcp_rpn[i] * Q_dot_rn_n[i - 1]
    T_r4n[i] = T_r4n[i - 1] + ts / tau_hx_r * (T_rp_n[i - 1] - T_r4n[i - 1]) + ts / mcp_rpn[i] * Q_dot_rn_n[i - 1]

    Q_dot_rsn_n[i] = hA_rsn * (T_r4n[i - 1] - T_rs_n[i - 1])  # MW - TDAMSRE p.7
    # mass flow
    mcp_rsn[i] = mn_rs * scp_rs  # (mass of material x heat capacity of material) of coolant salt per lump in MW-s/°C
    # new temperatures
    T_rs_n[i] = T_rs_n[i - 1] + ts / tau_hx_r * (T_rsn[i - 1] - T_rs_n[i - 1]) + ts / mcp_rsn[i] * Q_dot_rsn_n[i - 1]
    T_rsn[i] = T_rsn[i - 1] + ts / tau_hx_r * (T_rs_n[i - 1] - T_rsn[i - 1]) + ts / mcp_rsn[i] * Q_dot_rsn_n[i - 1]

    # Output power from the core
    Power_n[i] = beta_eff(ieff) * (rho_0 - beta_t) * n_frac_n[i] * Lam / n_frac_ss + P
    # Calculate the feedback reactivity
    feedback_rho[i] = (rho_0 - rho_0_0) / rho_0_0

# Plot the results
plt.figure()
plt.plot(t / 3600, Power_n, label='Core Power')
plt.xlabel('Time (hours)')
plt.ylabel('Core Power (MW)')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t / 3600, T_p1n, label='Primary Inlet')
plt.plot(t / 3600, T_p2n, label='Fuel Exit')
plt.plot(t / 3600, T_p3n, label='Fertile Exit')
plt.plot(t / 3600, T_p4n, label='Primary Outlet')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t / 3600, T_s1n, label='Secondary Inlet')
plt.plot(t / 3600, T_s2n, label='Hot Outlet')
plt.plot(t / 3600, T_s3n, label='Cold Outlet')
plt.plot(t / 3600, T_s4n, label='Secondary Outlet')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t / 3600, T_rp_n, label='Radiator Inlet')
plt.plot(t / 3600, T_r4n, label='Radiator Outlet')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t / 3600, n_frac_n, label='Normalized Fraction')
plt.xlabel('Time (hours)')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t / 3600, feedback_rho, label='Reactivity Feedback')
plt.xlabel('Time (hours)')
plt.ylabel('Reactivity Feedback')
plt.legend()
plt.grid(True)

plt.show()
