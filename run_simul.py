"""
Run simulation.
"""

import os
from pathlib import Path
import main
import pickle
import util

params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 128,      # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 10,         # synaptic delay in the network, in ms
    'n_pat': 16,         # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 1e-2,         # learning rate
    'n_trial': 1e4,      # number of trials
    't_dur': 2,          # duration of trial
    'train': True,       # whether to train network or not
    'W_rec': None,       # recurrent weights of associative network
    'W_ff': None,        # feedforward weights to associative neurons
    'W_fb': None,        # feedback weights to associative neurons
    'S': None,           # sign of neurons
    'US': None,          # set of US inputs
    'CS': None,          # set of CS inputs
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': False        # whether the network respects Dale's law
    }

# Save directory
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
filename = util.filename(params)

# Run simulation
net = main.network(params)
net.simulate()
net.est_US()

# Save results
with open(data_path + filename + '.pkl','wb') as f:
    pickle.dump(net, f)