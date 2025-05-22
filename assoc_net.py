"""
Functions that set up the associative network.
"""

import numpy as np
import util

# Define dynamics of associative network

def dynamics(r,I_ff,I_fb,W_rec,W_ff,W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,dt,
             n_sigma,g_sh,I_inh,fun,a,tau_s,tau_l=20,gD=.2,gL=.1,E_e=14/3,E_i=-1/3):
    
    # units in ms or ms^-1
    n_neu = np.size(r); c = gD/gL
    
    # Create noise that will be added to all origins of input
    n = np.random.normal(0,n_sigma,n_neu)
    n_d = np.random.normal(0,n_sigma,n_neu)
    
    # input to the dendrites
    I_d += (- I_d + np.dot(W_rec,r) + np.dot(W_fb,I_fb) + I_inh + n_d)*dt/tau_s
    
    # Dentritic potential is a low-pass filtered version of the dentritic current
    V_d += (-V_d+I_d)*dt/tau_l
    
    # Time-dependent somatic conductances
    g_e += (-g_e + np.dot(W_ff.clip(min=0),I_ff))*dt/tau_s
    g_i += (-g_i - np.dot(W_ff.clip(max=0),I_ff))*dt/tau_s
    
    # Input to the soma (teacher signal)
    I = g_e*(E_e-V) + (g_i+g_sh)*(E_i-V)
    
    # Somatic voltage
    V += (-V + c*(V_d-V) + I/gL + n)*dt/tau_l
    
    # Firing rate
    r = util.act_fun(V,fun)
    
    # Dendritic prediction of somatic voltage
    V_ss = a*V_d*gD/(gD+gL)
    
    # Discrepancy between dendritic prediction and actual firing rate
    error = r - util.act_fun(V_ss,fun)
    
    # Compute PSP for every dendritic input to associative neurons
    r_in = np.concatenate((r,I_fb))
    I_PSP += (- I_PSP + r_in)*dt/tau_s
    PSP += (- PSP + I_PSP)*dt/tau_l
    
    return r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i


# Dopamine uptake and release dynamics

def neuromodulator_dynamics(C_p_u,C_p_r,C_n_u,C_n_r,S,dt,tau_r=200,tau_u=300):
    
    # Neurotransmitter released with surprise only
    if S != 0:
        C_p_r += max(S,0) * 1e3/tau_r
        C_n_r += max(-S,0) * 1e3/tau_r
    
    # Amount of available neuromodulator decays with time
    C_p_r += -C_p_r * dt/tau_r
    C_n_r += -C_n_r * dt/tau_r
    
    # Amount of neuromodulator uptake lags behind available concentration
    C_p_u += (-C_p_u + C_p_r) * dt/tau_u
    C_n_u += (-C_n_u + C_n_r) * dt/tau_u
    
    return C_p_u, C_p_r, C_n_u, C_n_r


# Learning rate as a function of neuromodulator uptake
    
def learn_rate(C_p_u,C_n_u,eta):
    return eta * (C_p_u - C_n_u)


# Define learning rule dynamics

def learn_rule(W_rec,W_fb,r,error,Delta,PSP,eta,dt,dale,sign,filt=False,
               rule='Pred',norm=10,r_m=.02,tau_d=100,no_recurrent=False):
    
    n_neu = W_rec.shape[0]
    
    # Weight update
    if rule == 'Pred':
        PI = np.outer(error,PSP)
    elif rule == 'Hebb':
        W = np.concatenate((W_rec,W_fb),axis=1)
        PI = np.outer(r,PSP) - norm*np.multiply(r[:,np.newaxis]**2,W)
    elif rule == 'BCM':
        PI = np.outer(r*(r-r_m),PSP)
    elif rule == 'W_decay':
        W = np.concatenate((W_rec,W_fb),axis=1)
        PI = np.outer(r,PSP) - norm*W
    
    # Low-pass filter weight updates
    if filt:
        Delta += (PI - Delta)*dt/tau_d
    else:
        Delta = PI
    dW = eta*Delta*dt
    
    # Separate matrices
    dW_rec, dW_fb = np.split(dW,[n_neu],axis=1)
    
    # Perform weight updates
    if not no_recurrent:
        W_rec += dW_rec
    W_fb += dW_fb
    
    # Set every weight that violates Dale's law to zero
    if dale:
        W_rec[np.dot(W_rec,sign)<0] = 0
    
    return W_rec, W_fb, dW_rec, dW_fb


# Find instructed steady-state firing rate for given feedforward input

def ss_fr(I_ff,W_ff,g_inh,fun,E_e=14/3,E_i=-1/3):
    
    # Steady-state somatic conductances
    g_e = np.dot(W_ff.clip(min=0),I_ff)
    g_i = - np.dot(W_ff.clip(max=0),I_ff)
    
    # Matching potential (equilibrium)
    V_m = (g_e*E_e+(g_i+g_inh)*E_i)/(g_e+g_i+g_inh)
    
    # Teacher-imposed firing rate
    r_m = util.act_fun(V_m,fun)
    
    return r_m