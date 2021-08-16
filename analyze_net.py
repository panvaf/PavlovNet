"""
Analyze trained network.
"""

import os
from pathlib import Path
import pickle
import util
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import main
import torch

# Load network
n_CS = 1

params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_mem': 64,         # number of memory neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_pat': 1,         # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 5e-4,         # learning rate
    'n_trial': 1e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_disap': 2,       # time in trial that CS disappears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'W_rec': None,       # recurrent weights of associative network
    'W_ff': None,        # feedforward weights to associative neurons
    'W_fb': None,        # feedback weights to associative neurons
    'US': None,          # set of US inputs
    'CS': None,          # set of CS inputs
    'R': None,           # reward associated with every US
    'S': None,           # sign of neurons
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': False,       # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': None,  # Memory RNN to load
    'out': False,        # whether to feed output of RNN to associative net
    'est_every': True,   # whether to estimate US and reward after every trial
    'flip': False,       # whether to flip the US-CS associations mid-learning
    'exact': False,      # whether to demand an exact Hamming distance between patterns
    'low': .5,            # lowest possible reward
    'filter': True      # whether to filter the learning dynamics
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
    'overexp': True,     # whether to test for overexpectation effects
    'salience': 1        # relative saliance of CSs
    }

data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
if n_CS == 1:    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DAreprod'
elif n_CS == 2:
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20DA'

with open(data_path+filename+'.pkl', 'rb') as f:
    net = pickle.load(f)


# Fontsize appropriate for plots
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title

if n_CS == 1:
    
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
    
    
    # Scatterplot of US- and CS- induced firing rates
    
    fig, ax = plt.subplots(figsize=(1.5,1.5))
    ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0)
    ax.scatter(net.Phi.flatten()*1000,net.Phi_est.flatten()*1000,s=.25,color='green',alpha=.5,zorder=1)
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    ax.set_xlabel('$f(\mathbf{V}) \|_{US}$ (spikes/s)')
    ax.set_ylabel('$f(\mathbf{V}) \|_{CS}$ (spikes/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', -5))
    ax.spines['bottom'].set_position(('data', -5))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    
    
    # Scatterplot of actual and decoded US digits
    
    fig, ax = plt.subplots(figsize=(1.5,1.5))
    ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0)
    ax.scatter(net.US.flatten()+np.random.normal(scale=.02,size=np.size(net.US)),net.US_est.flatten(),s=.25,color='green',alpha=.5,zorder=1)
    ax.set_xlim([-.2,1.2])
    ax.set_ylim([-.2,1.2])
    ax.set_xlabel('$\mathbf{r}^{US}$ digit')
    ax.set_ylabel('$\hat{\mathbf{r}}^{US}_{opt}$ digit')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', -.25))
    ax.spines['bottom'].set_position(('data', -.25))
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.xaxis.set_minor_locator(MultipleLocator(.25))
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_minor_locator(MultipleLocator(.25))


    # Short-term memory leak plot
    
    if params['mem_net_id'] is not None:
        # Load memory net
        mem_net = main.RNN(params['n_in'],params['n_mem'],params['n_in'],params['n_sigma'],
                       params['tau_s'],params['dt']*1e3)
        checkpoint = torch.load(data_path + params['mem_net_id'] + '.pth')
        mem_net.load_state_dict(checkpoint['state_dict'])
        mem_net.eval()
        
        # Input
        t_CS = .5; t_trial = 3
        n_CS = int(t_CS/params['dt']); n_trial = int(t_trial/params['dt'])
        t = np.linspace(0,3,n_trial)
        torch.manual_seed(40)
        CS = torch.randint(low=0, high=2,size=(1,1,params['n_in']))
        inp = CS.repeat(1,n_trial,1).float()
        inp[:,n_CS:,:] = 0
        
        out, _ = mem_net(inp)
        CS_est = out[0, :, :].detach().numpy()
        
        # Plot
        t_skip = .2; n_skip = int(t_skip/params['dt'])
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(t[n_skip:],CS_est[n_skip:,:],linewidth=1,zorder=1)
        ax.axvline(x=.5,color='black',linestyle='dotted',linewidth=1.5,label='$CS$ removed',zorder=0)
        ax.set_xlim([t_skip,t_trial])
        ax.set_ylim([-.2,1.5])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$\hat{\mathbf{r}}^{CS}$ digit')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', t_skip -.15))
        ax.spines['bottom'].set_position(('data', -.25))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(.5))
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        fig.legend(frameon=False)


# Reward conditioning plot

if net.est_every:
    R = net.R
    n_trial = net.n_trial
    
    if n_CS == 1:
        R_est = net.R_est
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(R_est,label='$\hat{R}$',c='green')
        ax.axhline(y=net.R,c='black',linestyle='--',label='$R$')
        ax.set_xlabel('Trials')
        ax.set_ylabel('Reward')
        ax.set_xlim([0,n_trial])
        ax.set_ylim([-.02*R,1.02*R])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.05*n_trial))
        ax.spines['bottom'].set_position(('data', -.05*R))
        ax.xaxis.set_major_locator(MultipleLocator(int(n_trial/2)))
        ax.xaxis.set_minor_locator(MultipleLocator(int(n_trial/4)))
        ax.yaxis.set_major_locator(MultipleLocator(R/2))
        ax.yaxis.set_minor_locator(MultipleLocator(R/4))
        fig.legend(frameon=False,ncol=2)
        
    elif n_CS == 2:
        R_est_1 = net.R_est_1
        R_est_2 = net.R_est_2
        R_est = R_est_1 + R_est_2
        R_est_max = np.max(R_est); R_max = np.max([R_est_max,R])
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(R_est_1,label='$\hat{R}_1$',c='dodgerblue')
        ax.plot(R_est_2,label='$\hat{R}_2$',c='darkorange')
        ax.plot(R_est,label='$\hat{R}$',c='green')
        ax.axhline(y=net.R,c='black',linestyle='--',label='$R$')
        ax.set_xlabel('Trials')
        ax.set_ylabel('Reward')
        ax.set_xlim([0,n_trial])
        ax.set_ylim([-.02*R_max,1.02*R_max])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.05*n_trial))
        ax.spines['bottom'].set_position(('data', -.05*R_max))
        ax.xaxis.set_major_locator(MultipleLocator(int(n_trial/2)))
        ax.xaxis.set_minor_locator(MultipleLocator(int(n_trial/4)))
        ax.yaxis.set_major_locator(MultipleLocator(R_max/2))
        ax.yaxis.set_minor_locator(MultipleLocator(R_max/4))
        fig.legend(frameon=False,ncol=4)

    #plt.savefig('3b.png',bbox_inches='tight',format='png',dpi=300)