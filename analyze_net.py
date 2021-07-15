"""
Analyze trained network.
"""

import os
from pathlib import Path
import pickle
import util
import matplotlib.pyplot as plt
import numpy as np

# Load network
n_CS = 2

params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_mem': 64,         # number of memory neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_pat': 16,         # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 5e-2,         # learning rate
    'n_trial': 1e3,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_disap': 2,      # time in trial that CS disappears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'W_rec': None,       # recurrent weights of associative network
    'W_ff': None,        # feedforward weights to associative neurons
    'W_fb': None,        # feedback weights to associative neurons
    'US': None,          # set of US inputs
    'CS': None,          # set of CS inputs
    'R': None,           # reward associated with every US
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': False,       # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': 'MemNet64tdur3iter1e5Noise0.1',  # Memory RNN to load
    'out': True,         # whether to feed output of RNN to associative net
    'est_every': False   # whether to estimate US and reward after every trial
    }

params2 = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_in': 20,          # size of patterns
    'eta': 5e-4,         # learning rate
    'n_trial': 3e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_2_ap_tr': 1e2,   # trial number in which CS 2 appears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': False,       # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'est_every': True,   # whether to estimate US and reward after every trial
    'overexp': True      # whether to test for overexpectation effects
    }

data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
if n_CS == 1:    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DA'
elif n_CS == 2:
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20DA'

with open(data_path+filename+'.pkl', 'rb') as f:
    net = pickle.load(f)


# Plot steady-state firing rate and decoding errors

err = net.Phi_est - net.Phi
dec_err = net.US_est - net.US

plt.hist(err.flatten()*1000,100)
plt.xlabel('Error (spikes/s)')
plt.ylabel('Count')
plt.title('Difference btw predicted and instructed firing rates')
plt.show()

plt.hist(net.Phi.flatten()*1000,100)
plt.xlabel('Firing rate (spikes/s)')
plt.ylabel('Count')
plt.title('Instructed firing rates')
plt.show()

plt.hist(dec_err.flatten(),100)
plt.xlabel('Error')
plt.ylabel('Count')
plt.title('Binary digit decoding error')
plt.show()

print('Average bit error per pattern is {} bits'.format(round(np.mean(np.sqrt(np.sum(dec_err**2,1))),2)))