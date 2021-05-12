"""
Run simulation.
"""

import main

params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 128,      # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 65,         # synaptic delay in the network, in ms
    'n_pat': 16,         # number of US/CS pattern associations to be learned
    'n_in': 10,          # size of patterns
    'H_d': 4,            # minimal acceptable Hamming distance between patterns
    'eta': 5e-2,         # learning rate
    'n_trial': 1e4,      # number of trials
    't_dur': 2,          # duration of trial
    'train': True,       # whether to train network or not
    'W_rec': None,       # recurrent weights of associative network
    'W_ff': None,        # feedforward weights to associative neurons
    'W_fb': None,        # feedback weights to associative neurons
    'US': None,          # set of US inputs
    'CS': None,          # set of CS inputs
    'fun': 'logistic'    # activation function of associative network
    }

sim = main.simulation(params)
sim.est_decoder()
sim.simulate()