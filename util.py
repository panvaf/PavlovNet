"""
Utilities.
"""

import numpy as np
from scipy.spatial.distance import hamming
import torch


# Choose activation function

def act_fun(x,fun):
    
    if fun == 'rect':
        r = rect(x)
    elif fun == 'logistic':
        r = logistic(x)
    
    return r


# Logistic function
    
def logistic(x,x0=1.5,b=2,s=.1):
    # s: maximum firing rate in kHz
    # x0: 50 % firing rate point
    # b: steepness of gain
    
    return s/(1+np.exp(-b*(x-x0)))


# Rectification function

def rect(x,beta=50):
    x[x<0] = 0
    return beta*x


# Create sets of USs and corresponding CSs

def gen_US_CS(n_pat,n_in,H_d):
    # n_pat: number of US-CS patterns we want to associate
    # n_in: size of binary number representing each pattern
    # H_d: minimal acceptable Hamming distance between any two patterns of
    #      the same type. Current algo is simple, but effective. Could maximize
    #      distance with Reed-Solomon code, however not very biological..
    
    patterns = np.empty((2,n_pat,n_in))
    
    for st in range(2):
        for i in range(n_pat):
            while True:
                sw = 0
                patt = np.random.choice([0,1],n_in)
                # Make sure Hamming distance with existing patterns is acceptable.
                # Algo is greedy, not trying to spread codewords evenly.
                for j in range(i):
                    h_d = hamming(patt,patterns[st,j,:])*n_in
                    if h_d < H_d:
                        # Patterns too close
                        sw = 1
                        break
                # If new pattern is spaced apart from others, break while loop
                if not sw:
                    break
            patterns[st,i,:] = patt
            
    US = patterns[0,:]
    CS = patterns[1,:]
    
    return US, CS


# Returns filename from network parameters

def filename(params):
    
    filename =  format(params['n_trial'],'.0e').replace('+0','') + \
        'trials' + str(params['n_pat']) + 'pat' + \
        (('tdur' + str(params['t_dur'])) if params['t_dur'] != 2 else '') + \
        (('CSdis' + str(params['CS_disap'])) if params['CS_disap'] != params['t_dur'] else '') + \
        (('USap' + str(params['US_ap'])) if params['US_ap'] != 0 else '') + \
        (('insz' + str(params['n_in'])) if params['n_in'] != 20 else '') + \
        (('Hd' + str(params['H_d'])) if params['H_d'] != 8 else '') + \
        (('taus' + str(params['tau_s'])) if params['tau_s'] != 10 else '') + \
        (('inh' + str(params['I_inh'])) if params['I_inh'] else '') + \
        (('n' + str(params['n_sigma']).replace(".","")) if params['n_sigma'] != 0 else '') + \
        (('N' + str(params['n_assoc'])) if params['n_assoc'] != 128 else '') + \
        (('eta' + str(params['eta'])) if params['eta'] != 1e-2 else '') + \
        ('Dale' if params['dale'] else '') + \
        ('MemNet' if params['mem_net_id'] is not None else '') + \
        ('Out' if params['out'] else '') + ('EstEv' if params['est_every'] else '')
        
    return filename


def filename2(params):
    
    filename =  format(params['n_trial'],'.0e').replace('+0','') + 'trials' + \
        (('tdur' + str(params['t_dur'])) if params['t_dur'] != 2 else '') + \
        (('USap' + str(params['US_ap'])) if params['US_ap'] != 0 else '') + \
        (('CS2ap' + str(params['CS_2_ap_tr'])) if params['CS_2_ap_tr'] != 0 else '') + \
        (('insz' + str(params['n_in'])) if params['n_in'] != 20 else '') + \
        (('taus' + str(params['tau_s'])) if params['tau_s'] != 10 else '') + \
        (('inh' + str(params['I_inh'])) if params['I_inh'] else '') + \
        (('n' + str(params['n_sigma']).replace(".","")) if params['n_sigma'] != 0 else '') + \
        (('N' + str(params['n_assoc'])) if params['n_assoc'] != 64 else '') + \
        (('eta' + str(params['eta'])) if params['eta'] != 1e-2 else '') + \
        ('Dale' if params['dale'] else '') + ('EstEv' if params['est_every'] else '') + \
        ('Overexp' if params['overexp'] else '')
        
    return filename


# Weighted mean squared error loss

def MSELoss_weighted(output,target,mask):
    loss = torch.sum(mask*(output - target)**2)
    size = torch.numel(target)
    norm = torch.sum(mask)
    
    avg_loss = loss/size
    normed_loss = loss/norm
    
    return avg_loss, normed_loss