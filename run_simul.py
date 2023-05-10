"""
Run simulations and store results.
"""

import os
from pathlib import Path
import main
import pickle
import util
import numpy as np

# Which network and how many CS-US associations to run
n_CS = 1
n_pat = 1

# Whether to use the same network initialization and CS-US sets
reprod = True
if reprod:
    d = np.load('reproduce1.npz'); W_rec = d['W_rec']; W_ff = d['W_ff']; S = np.diag(np.ones(W_rec.shape[0]))
    W_fb = d['W_fb']; US = d['US'][0:n_pat]; CS = d['CS'][0:n_pat]; R = np.ones(W_rec.shape[0])[0:n_pat]
else:
    W_rec = None; W_ff = None; W_fb = None; US = None; CS = None; R = None; S = None

params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 32,       # number of associative neurons
    'n_mem': 64,         # number of memory neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_pat': n_pat,      # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 5e-3,         # learning rate
    'a': .97,              # deviation from self-consistency
    'n_trial': 50,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_disap': 2,       # time in trial that CS disappears
    'US_ap': 1,          # time in trial that US appears
    'US_jit': 0,         # random jitter in the time that the US appears
    'train': True,       # whether to train network or not
    'W_rec': W_rec,      # recurrent weights of associative network
    'W_ff': W_ff,        # feedforward weights to associative neurons
    'W_fb': W_fb,        # feedback weights to associative neurons
    'US': US,            # set of US inputs
    'CS': CS,            # set of CS inputs
    'R': R,              # reward associated with every US
    'S': S,              # sign of neurons
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 2,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': 'MemNet64tdur3iter1e5Noise0.1',  # Memory RNN to load
    'out': True,         # whether to feed output of RNN to associative net
    'est_every': True,  # whether to estimate US and reward after every trial
    'DA_plot': True,    # whether to keep track of expected reward within trial
    'trial_dyn': True,  # whether to store trial dynamics
    'GiveR': True,       # whether to provide reward upon US presentation
    'flip': False,       # whether to flip the US-CS associations mid-learning
    'extinct': True,    # whether to undergo extinction of learned associations
    'reacquire': False,  # whether to undergo extinction and reacquisition of learned association
    'exact': False,      # whether to demand an exact Hamming distance between patterns
    'low': 1,            # lowest possible reward
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'run': 0,            # number of run for many runs of same simulation
    'm': 2               # order of gaussian for radial basis function
    }

params2 = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_in': 20,          # size of patterns
    'eta': 5e-4,         # learning rate
    'a': 1,           # deviation from self-consistency
    'n_trial': 5e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_2_ap_tr': 0,     # trial number in which CS 2 appears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'est_every': True,   # whether to estimate US and reward after every trial
    'overexp': False,    # whether to test for overexpectation effects
    'salience': 1,       # relative salience of CSs
    'cont': [.8,.4],       # contingencies of CSs
    'cond_dep': True, # whether one CS is conditionally dependent on the other
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,         # normalization strenght for learning rule
    'm': 2               # order of gaussian for radial basis function
    }

# Save directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'trained_networks')
if n_CS == 1:    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DAOnline' + ('reprod' if reprod else '')
elif n_CS == 2:
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20DAOnline'

# Run simulation
if n_CS == 1:    
    net = main.network(params)
elif n_CS == 2:
    net = main.network2(params2)

net.simulate()

# Save results
with open(os.path.join(data_path,filename + '.pkl'),'wb') as f:
    pickle.dump(net, f)