"""
Analyze trained network.
"""

import os
from pathlib import Path
import pickle
import util
import matplotlib.pyplot as plt


# Load network

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
    'US': None,          # set of US inputs
    'CS': None,          # set of CS inputs
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1      # store errors this often
    }

data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
filename = util.filename(params)

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