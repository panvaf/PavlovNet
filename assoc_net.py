"""
Functions that set up the associative network.
"""

import numpy as np
import util

# Define dynamics of associative network

def dynamics(r,I_ff,I_fb,W_rec,W_ff,W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,dt,
             n_sigma,g_sh,I_inh,fun,tau_s,tau_l=20,gD=.2,gL=.1,E_e=14/3,E_i=-1/3):
    
    # units in ms or ms^-1
    n_neu = np.size(r); c = gD/gL
    
    # Create noise that will be added to all origins of input
    n = np.random.normal(0,n_sigma,n_neu)
    n_d = np.random.normal(0,n_sigma,n_neu)
    
    # input to the dendrites
    I_d += (- I_d + np.dot(W_fb,I_fb) + I_inh + n_d)*dt/tau_s
    
    # Dentritic potential is a low-pass filtered version of the dentritic current
    V_d += (-V_d+I_d)*dt/tau_l
    
    # Time-dependent somatic conductances
    g_e += (-g_e + np.dot(W_ff.clip(min=0),I_ff))*dt/tau_s
    g_i += (-g_i - np.dot(W_ff.clip(max=0),I_ff))*dt/tau_s
    
    # Input to the soma (teacher signal)
    I = g_e*(E_e-V) + (g_i+g_sh)*(E_i-V)
    
    # Somatic voltage
    V += (-V + c*(V_d-V) + I/gL + n)*dt/tau_l
    
    r = util.act_fun(V,fun)
    # Strong coupling of the dentrite to the soma
    V_ss = V_d*gD/(gD+gL)
    error = r - util.act_fun(V_ss,fun)

    # Compute PSP for every input to associative neurons
    r_in = np.concatenate((r,I_fb))
    I_PSP += (- I_PSP + r_in)*dt/tau_s
    PSP += (- PSP + I_PSP)*dt/tau_l
    
    return r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i


# Define learning rule dynamics

def learn_rule(W_rec,W_fb,error,Delta,PSP,eta,dt,dale,S,tau_d=100):
    
    n_neu = W_rec.shape[0]
    
    # Weight update
    PI = np.outer(error,PSP)
    Delta += (PI - Delta)*dt/tau_d
    dW = eta*Delta*dt
    
    # Separate matrices
    dW_rec, dW_fb = np.split(dW,[n_neu],axis=1)
    #W_rec += dW_rec
    W_fb += dW_fb
    
    # Set every weight that violates Dale's law to zero
    if dale:
        W_rec[np.dot(W_rec,S)<0] = 0
    
    return W_rec, W_fb


# Find instructed steady-state firing rate for given feedforward input

def ss_fr(I_ff,W_ff,g_sh,fun,E_e=14/3,E_i=-1/3):
    
    # Steady-state somatic conductances
    g_e = np.dot(W_ff.clip(min=0),I_ff)
    g_i = - np.dot(W_ff.clip(max=0),I_ff)
    
    # Matching potential (equilibrium)
    V_m = (g_e*E_e+(g_i+g_sh)*E_i)/(g_e+g_i+g_sh)
    
    # Teacher-imposed firing rate
    r_m = util.act_fun(V_m,fun)
    
    return r_m