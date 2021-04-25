"""
Functions that set up the associative network.
"""

import numpy as np
import util


# Define dynamics of associative network

def dynamics(r,in_ff,in_fb,w_rec,w_ff,w_fb,V,I_d,V_d,PSP,I_PSP,dt,n_sigma,tau_s=65,tau_l=10,gD=2,gL=1):
    
    # units in ms or ms^-1, C is considered unity and the unit is embedded in g
    n_neu = np.size(r)
    
    # Create noise that will be added to all origins of input
    N = np.random.normal(0,n_sigma,n_neu)
    N_d = np.random.normal(0,n_sigma,n_neu)
        
    # input to the dendrites
    I_d += (- I_d + np.dot(w_rec,r) + np.dot(w_fb,in_fb) + N_d) * dt/tau_s
    
    # Dentritic potential is a low-pass filtered version of the dentritic current
    V_d += (-V_d+I_d)*dt/tau_l
    
    # input to the soma (teacher signal)
    V += (-gL*V + gD*(V_d-V) + np.dot(w_ff,in_ff) + N)*dt
    
    r = util.logistic(V)
    # Strong coupling of the soma to the dentrite
    V_ss = V_d*gD/(gD+gL)
    error = r - util.logistic(V_ss)

    # Compute PSP for every input to associative neurons
    r_in = np.concatenate((r,in_fb))
    I_PSP += (- I_PSP + r_in) * dt/tau_s
    PSP += (-PSP+I_PSP) * dt/tau_l
            
    return r, V, I_d, V_d, error, PSP, I_PSP


# Define learning rule dynamics

def learn_rule(w_rec,w_fb,error,Delta,PSP,eta,dt,tau_d=100):
    
    n_neu = w_rec.shape[0]
    
    # Weight update
    PI = np.outer(error,PSP)
    Delta += (PI - Delta)*dt/tau_d
    dw = eta*Delta*dt
            
    # Separate matrices
    dw_rec, dw_fb = np.split(dw,[n_neu],axis=1)
    w_rec += dw_rec
    w_fb += dw_fb
    
    return w_rec, w_fb