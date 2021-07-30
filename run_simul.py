"""
Run simulation.
"""

import os
from pathlib import Path
import main
import pickle
import util
import numpy as np

n_CS = 1
reprod = True
n_pat = 1

if reprod:
    d = np.load('reproduce.npz'); W_rec = d['W_rec']; W_ff = d['W_ff']
    W_fb = d['W_fb']; US = d['US'][0:n_pat]; CS = d['CS'][0:n_pat]; R = d['R'][0:n_pat]
else:
    W_rec = None; W_ff = None; W_fb = None; US = None; CS = None; R = None

params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_mem': 64,         # number of memory neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_pat': n_pat,      # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 5e-4,         # learning rate
    'n_trial': 1e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_disap': 2,       # time in trial that CS disappears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'W_rec': W_rec,      # recurrent weights of associative network
    'W_ff': W_ff,        # feedforward weights to associative neurons
    'W_fb': W_fb,        # feedback weights to associative neurons
    'US': US,            # set of US inputs
    'CS': CS,            # set of CS inputs
    'R': R,              # reward associated with every US
    'S': None,           # sign of neurons
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': False,       # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': None,  # Memory RNN to load
    'out': True,         # whether to feed output of RNN to associative net
    'est_every': True    # whether to estimate US and reward after every trial
    }

params2 = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_in': 20,          # size of patterns
    'eta': 5e-4,         # learning rate
    'n_trial': 1e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_2_ap_tr': 0,     # trial number in which CS 2 appears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': False,       # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'est_every': True,   # whether to estimate US and reward after every trial
    'overexp': False,    # whether to test for overexpectation effects
    'salience': 1        # relative saliance of CSs
    }

# Save directory
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
if n_CS == 1:    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DA' + ('reprod' if reprod else '')
elif n_CS == 2:
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20DA'

# Run simulation
if n_CS == 1:    
    net = main.network(params)
elif n_CS == 2:
    net = main.network2(params2)

net.simulate()

# Save results
with open(data_path + filename + '.pkl','wb') as f:
    pickle.dump(net, f)