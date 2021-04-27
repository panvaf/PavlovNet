"""
Functions that set up the associative network.
"""

import numpy as np
import util

# Define dynamics of associative network

def dynamics(r,I_ff,I_fb,W_rec,W_ff,W_fb,V,I_d,V_d,PSP,I_PSP,dt,n_sigma,exc,
                                         fun,tau_s=65,tau_l=10,gD=2,gL=1):
    
    # units in ms or ms^-1, C is considered unity and the unit is embedded in g
    n_neu = np.size(r)
    
    # Create noise that will be added to all origins of input
    N = np.random.normal(0,n_sigma,n_neu)
    N_d = np.random.normal(0,n_sigma,n_neu)
        
    # input to the dendrites
    I_d += (- I_d + np.dot(W_rec,r) + np.dot(W_fb,I_fb) + N_d) * dt/tau_s
    
    # Dentritic potential is a low-pass filtered version of the dentritic current
    V_d += (-V_d+I_d)*dt/tau_l
    
    # input to the soma (teacher signal)
    V += (-gL*V + gD*(V_d-V) + np.dot(W_ff,I_ff) + exc + N)*dt
    
    r = util.act_fun(V,fun)
    # Strong coupling of the soma to the dentrite
    V_ss = V_d*gD/(gD+gL)
    error = r - util.act_fun(V_ss,fun)

    # Compute PSP for every input to associative neurons
    r_in = np.concatenate((r,I_fb))
    I_PSP += (- I_PSP + r_in) * dt/tau_s
    PSP += (-PSP+I_PSP) * dt/tau_l
    
    return r, V, I_d, V_d, error, PSP, I_PSP


# Define learning rule dynamics

def learn_rule(W_rec,W_fb,error,Delta,PSP,eta,dt,tau_d=100):
    
    n_neu = W_rec.shape[0]
    
    # Weight update
    PI = np.outer(error,PSP)
    Delta += (PI - Delta)*dt/tau_d
    dW = eta*Delta*dt
    
    # Separate matrices
    dW_rec, dW_fb = np.split(dW,[n_neu],axis=1)
    W_rec += dW_rec
    W_fb += dW_fb
    
    return W_rec, W_fb