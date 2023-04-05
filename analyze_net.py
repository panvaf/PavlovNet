"""
Analyze trained network and produce figures.
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
from mpl_toolkits.mplot3d import axes3d

# Which network type to load
n_CS = 1

# Determine parameters to load the appropriate network
params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_mem': 64,         # number of memory neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_pat': 1,         # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 2e-4,         # learning rate
    'a': .01,              # deviation from self-consistency
    'n_trial': 1e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_disap': 2,       # time in trial that CS disappears
    'US_ap': 1,          # time in trial that US appears
    'US_jit': 0,         # random jitter in the time that the US appears
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
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': 'MemNet64tdur3iter1e5Noise0.1',  # Memory RNN to load
    'out': True,         # whether to feed output of RNN to associative net
    'est_every': True,  # whether to estimate US and reward after every trial
    'DA_plot': False,    # whether to keep track of expected reward within trial
    'GiveR': True,       # whether to provide reward upon US presentation
    'flip': False,       # whether to flip the US-CS associations mid-learning
    'extinct': False,    # whether to undergo extinction of learned associations
    'reacquire': False,  # whether to undergo extinction and reacquisition of learned association
    'exact': False,      # whether to demand an exact Hamming distance between patterns
    'low': 1,            # lowest possible reward
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'run': 0,            # number of run for many runs of same simulation
    'm': 6               # order of gaussian for radial basis function
    }

params2 = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_in': 20,          # size of patterns
    'eta': 5e-4,         # learning rate
    'a': 0.97,           # deviation from self-consistency
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
    'salience': 1,       # relative saliance of CSs
    'cont': [.8,.4],       # contingencies of CSs
    'cond_dep': False,   # whether one CS is conditionally dependent on the other
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'm': 6               # order of gaussian for radial basis function
    }

# Load network
data_path = os.path.join(str(Path(os.getcwd()).parent),'trained_networks')
if n_CS == 1:    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DAonlinereprod'
elif n_CS == 2:
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20DAOnline'

with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
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
    
    # Obtain results
    Phi = net.Phi
    US = net.US
    if net.Phi_est.ndim==3:
        Phi_est_hist = net.Phi_est
        US_est_hist = net.US_est
        Phi_est = net.Phi_est[-1,:,:]
        US_est = net.US_est[-1,:,:]
    else:
        Phi_est = net.Phi_est
        US_est = net.US_est
    
    # Plot steady-state firing rate and decoding errors
    
    err = Phi_est - Phi
    dec_err = US_est - US
    
    plt.hist(err.flatten()*1000,100)
    plt.xlabel('Error (spikes/s)')
    plt.ylabel('Count')
    plt.title('Difference btw predicted and instructed firing rates')
    plt.show()
    
    plt.hist(Phi.flatten()*1000,100)
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
    ax.scatter(Phi.flatten()*1000,Phi_est.flatten()*1000,s=.25,
               color='green',alpha=.5,zorder=1)
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

    # Dopamine uptake plot
    if net.DA_plot:
        Z = net.DA_u
        x = np.linspace(0,net.t_dur,Z.shape[1])
        y = np.arange(net.n_trial)+1
        X, Y = np.meshgrid(x,y)
        
        x1 = 0.5*np.ones(y.shape); x1[0] = 0; x1[-1] = 0
        
        # Plot a 3D surface
        fig = plt.figure(figsize=(2.5,2.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(y, x1, zs=0, zdir='x', color='black', linewidth=2)
        ax.plot(y, x1, zs=3, zdir='x', color='black', linewidth=2)
        ax.plot_surface(X,Y,Z,rstride=1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trials')
        ax.set_zlabel('Dopamine Uptake')
        ax.set_xticks(np.linspace(0,net.t_dur,5))
        ax.set_yticks(np.linspace(0,len(y),5))
        ax.set_zticks([0,.5,1,1.5])
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)   
        ax.view_init(25, -105)
        plt.show()
    
    # Scatterplot of shaping history of CS-induced responses
    
    if net.Phi_est.ndim==3:
        
        snaps = np.array([0,2,9,49])
        trials = (snaps+1)/100 * net.n_trial; trials = trials.astype('int')
        cols = ['dodgerblue','darkslateblue','darkorange','green']
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0)
        for i, tr in enumerate(snaps): 
            ax.scatter(Phi.flatten()*1000,Phi_est_hist[tr,:].flatten()*1000,s=.25,
                       color=cols[i],alpha=.5,zorder=i+1,label='{}'.format(trials[i]))
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
        fig.legend(title='Trial #',frameon=False,ncol=1,bbox_to_anchor=(1.6, 1),
                   markerscale=3,title_fontsize=SMALL_SIZE)
        
        #plt.savefig('Sub_his.png',bbox_inches='tight',format='png',dpi=300)
        
        trials = net.n_trial*np.linspace(0,1,100/params['every_perc'])
        fig, ax = plt.subplots(figsize=(2,1.5))
        ax.plot(trials,net.R_est,linewidth=.5,alpha=.5)
        ax.plot(trials,np.average(net.R_est,axis=1),c='green',linewidth=1)
        ax.axhline(y=1,c='black',linestyle='--',linewidth=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.05*net.n_trial))
        ax.spines['bottom'].set_position(('data', -.05))
        ax.xaxis.set_major_locator(MultipleLocator(.5*net.n_trial))
        ax.xaxis.set_minor_locator(MultipleLocator(.25*net.n_trial))
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_xlim([0,net.n_trial])
        ax.set_ylabel('Reward')
        ax.set_xlabel('Trials')
        
        color = 'red'
        ax1 = ax.twinx()
        #ax1.plot(trials,np.var(US_est_hist-US,axis=2),color=color,linewidth=.5,alpha=.5)
        ax1.plot(trials,np.var(US_est_hist-US,axis=(1,2)),color=color,linewidth=1)
        ax1.set_ylabel('Var($\mathbf{r}^{US}$,$\hat{\mathbf{r}}^{US}\|_{CS}$)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['right'].set_position(('data', 1.05*net.n_trial))
        ax1.yaxis.set_major_locator(MultipleLocator(.1))
        ax1.set_ylim([0,.25])
        
        #plt.savefig('Cond_his.png',bbox_inches='tight',format='png',dpi=300)
    
    # Scatterplot of actual and decoded US digits
    
    fig, ax = plt.subplots(figsize=(1.5,1.5))
    ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0)
    ax.scatter(US.flatten()+np.random.normal(scale=.02,size=np.size(US)),
               US_est_hist[tr,:].flatten(),s=.25,color='green',alpha=.5,zorder=1)
    ax.set_xlim([-.2,1.2])
    ax.set_ylim([-.2,1.2])
    ax.set_xlabel('$\mathbf{r}^{US}$ element')
    ax.set_ylabel('$\hat{\mathbf{r}}^{US}\|_{CS}$ element')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', -.25))
    ax.spines['bottom'].set_position(('data', -.25))
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.xaxis.set_minor_locator(MultipleLocator(.25))
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_minor_locator(MultipleLocator(.25))
    
    #plt.savefig('USdec.png',bbox_inches='tight',format='png',dpi=300)


    # Short-term memory leak plot
    
    if params['mem_net_id'] is not None:
        # Load memory net
        mem_net = main.RNN(params['n_in'],params['n_mem'],params['n_in'],params['n_sigma'],
                       params['tau_s'],params['dt']*1e3)
        checkpoint = torch.load(os.path.join(data_path,params['mem_net_id'] + '.pth'))
        mem_net.load_state_dict(checkpoint['state_dict'])
        mem_net.eval()
        
        # Input
        t_CS = .5; t_trial = 3
        n_CS_disap = int(t_CS/params['dt']); n_trial = int(t_trial/params['dt'])
        t = np.linspace(0,3,n_trial)
        torch.manual_seed(40)
        CS = torch.randint(low=0, high=2,size=(1,1,params['n_in']))
        inp = CS.repeat(1,n_trial,1).float()
        inp[:,n_CS_disap:,:] = 0
        
        out, _ = mem_net(inp)
        CS_est = out[0, :, :].detach().numpy()
        
        # Plot
        t_skip = .2; n_skip = int(t_skip/params['dt'])
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(t[n_skip:],CS_est[n_skip:,:],linewidth=1,zorder=1)
        ax.axvspan(0,.5,facecolor='red',alpha=.2,label='$CS$ present',zorder=0)
        ax.set_xlim([t_skip,t_trial])
        ax.set_ylim([-.2,1.5])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$\hat{\mathbf{r}}^{CS}$ element')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', t_skip -.15))
        ax.spines['bottom'].set_position(('data', -.25))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(.5))
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        fig.legend(frameon=False)
        
        #plt.savefig('mem_leak.png',bbox_inches='tight',format='png',dpi=300)
        #plt.savefig('mem_leak.eps',bbox_inches='tight',format='eps',dpi=300)


# Reward conditioning plot

if net.est_every:
    R = net.R
    n_trial = net.n_trial
    
    if n_CS == 1:
        R_est = net.R_est
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(R_est,label='$\hat{R}$',c='green',linewidth=2)
        ax.axhline(y=R,c='black',linestyle='--',linewidth=2,label='$R$')
        #ax.axvline(x=10,linestyle='dotted',c='darkorange',linewidth=1.5,label='Extinction onset',zorder=0)
        ax.set_xlabel('Trials')
        ax.set_ylabel('Reward')
        ax.set_xlim([-.02*n_trial,n_trial])
        ax.set_ylim([-.02*R,1.02*R])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.05*n_trial))
        ax.spines['bottom'].set_position(('data', -.05*R))
        ax.xaxis.set_major_locator(MultipleLocator(int(n_trial/2)))
        ax.xaxis.set_minor_locator(MultipleLocator(n_trial/4))
        ax.yaxis.set_major_locator(MultipleLocator(R/2))
        ax.yaxis.set_minor_locator(MultipleLocator(R/4))
        fig.legend(frameon=False,loc='right')
        
    elif n_CS == 2:
        R_est_1 = net.R_est_1
        R_est_2 = net.R_est_2
        R_est = R_est_1 + R_est_2
        R_est_max = np.max(R_est); R_max = np.ceil(10*np.max([R_est_max,R]))/10; R_max = R
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(R_est_1,c='dodgerblue',linewidth=2,label='$\hat{R}_1$',zorder=1)
        ax.plot(R_est_2,c='darkorange',linewidth=2,label='$\hat{R}_2$',zorder=1)
        #ax.plot([],[],linestyle='',label='\n')
        #ax.axvline(x=100,linestyle='dotted',c='darkorange',linewidth=1.5,label='$CS_2$ presented',zorder=0)
        #ax.axvline(x=200,linestyle='dotted',c='green',linewidth=1.5,label='Both $CS$s presented',zorder=0)
        #ax.plot(R_est,c='green',linewidth=2,zorder=1)
        ax.axhline(y=R,c='black',linestyle='--',linewidth=2)
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
        fig.legend(frameon=False,loc='upper',ncol=2,bbox_to_anchor=(1.2, 1.35))
        
    #plt.savefig('Cond.png',bbox_inches='tight',format='png',dpi=300)
    #plt.savefig('Cond.eps',bbox_inches='tight',format='eps',dpi=300)