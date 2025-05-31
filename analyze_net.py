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
import matplotlib.lines as mlines

# Which network type to load
n_CS = 2

# Determine parameters to load the appropriate network
params = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_mem': 64,         # number of memory neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_pat': 16,         # number of US/CS pattern associations to be learned
    'n_in': 20,          # size of patterns
    'H_d': 8,            # minimal acceptable Hamming distance between patterns
    'eta': 5e-3,         # learning rate
    'a': 0.95,           # deviation from self-consistency
    'n_trial': 1e3,      # number of trials
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
    'sign': None,        # sign of neurons
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': 'MemNet64tdur3iter1e5Noise0.1',  # Memory RNN to load
    'out': True,         # whether to feed output of RNN to associative net
    'est_every': False,  # whether to estimate US and expectation after every trial
    'DA_plot': False,    # whether to keep track of expectation within trial
    'trial_dyn': False,  # whether to store trial dynamics
    'flip': False,       # whether to flip the US-CS associations mid-learning
    'extinct': False,    # whether to undergo extinction of learned associations
    't_wait': 5,         # time after US_ap that its considered an extinction trial
    'reacquire': False,  # whether to undergo extinction and reacquisition of learned association
    'exact': False,      # whether to demand an exact Hamming distance between patterns
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'T': 0.4,            # temporal window for averaging firing rates for BCM rule
    'run': 0,            # number of run for many runs of same simulation
    'm': 2,              # order of gaussian for radial basis function
    'no_recurrent': False # whether to disable recurrent weights
    }

params2 = {
    'dt': 1e-3,          # euler integration step size
    'n_assoc': 64,       # number of associative neurons
    'n_sigma': 0,        # input noise standard deviation
    'tau_s': 100,        # synaptic delay in the network, in ms
    'n_in': 20,          # size of patterns
    'eta': 5e-3,         # learning rate
    'a': 0.95,           # deviation from self-consistency
    'n_trial': 1e3,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_2_ap_tr': 0,     # trial number in which CS 2 appears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'est_every': False,   # whether to estimate US and expectation after every trial
    'overexp': False,    # whether to test for overexpectation effects
    'salience': 1,       # relative saliance of CSs
    'cont': [1,1],       # contingencies of CSs
    'cond_dep': False,   # whether one CS is conditionally dependent on the other
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'm': 2,              # order of gaussian for radial basis function
    'n_pat': 16,         # Number of US-CS patterns  
    'H_d': 8,            # Hamming distance between patterns
    'exact': False       # Exact Hamming distance constraint
    }

# Include regression lines in stimulus substitution plots
reg_lines = True

# Load network
data_path = os.path.join(str(Path(os.getcwd()).parent),'trained_networks')
if n_CS == 1:    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'
elif n_CS == 2:
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20'

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
    ax.set_ylim([0,102])
    ax.set_xlabel('$r^{us-only}_{rnn}$ (spikes/s)')
    ax.set_ylabel('$r^{cs-only}_{rnn}$ (spikes/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', -5))
    ax.spines['bottom'].set_position(('data', -5))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))

    # Feedforward and feedback weights, and difference of the two
    fig, axes = plt.subplots(1, 3, figsize=(6, 4))
    cmap = plt.cm.seismic
    cmap.set_bad('white')
    
    # Calculate symmetric limits for consistent scaling
    vmax = max(abs(net.W_ff).max(), abs(net.W_fb).max(), abs(net.W_ff - net.W_fb).max())
    vmin = -vmax  # Make it symmetric around 0

    # Feedback weights
    im = axes[0].imshow(net.W_fb, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Feedback Weights')
    axes[0].set_xlabel('Input Neuron')
    axes[0].set_ylabel('Output Neuron')
    
    # Feedforward weights
    im = axes[1].imshow(net.W_ff, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Feedforward Weights')
    
    # Difference of feedforward and feedback weights
    im = axes[2].imshow(net.W_ff - net.W_fb, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title('Difference')
  
    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.08, 0.03, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Weight')
    cbar.set_ticks([vmin, 0, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.1f}', '0', f'{vmax:.1f}'])

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()
    
    #plt.savefig('Stim_sub.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

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
    
    if net.Phi_est.ndim==3 and not net.est_every:
        
        snaps = np.array([0,2,9,49])
        trials = (snaps+1)/100 * net.n_trial * net.every_perc; trials = trials.astype('int')
        cols = ['dodgerblue','darkslateblue','darkorange','green']

        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0)

        legend_handles = []
        for i, tr in enumerate(snaps): 
            x = Phi.flatten()*1000
            y = Phi_est_hist[tr,:].flatten()*1000
            ax.scatter(x, y, s=.25, color=cols[i], alpha=.3, zorder=i+1)

            if reg_lines:
                # Calculate slope and intercept of regression line
                slope, intercept = np.polyfit(x, y, 1)
    
                # Plot regression line
                x_reg = np.array([x.min(), x.max()])
                y_reg = slope * x_reg + intercept
                ax.plot(x_reg, y_reg, color=cols[i], linestyle='-', linewidth=1.5, zorder=i+1)

            # Create custom legend handles
            legend_handles.append(mlines.Line2D([], [], marker='o', markersize=1, color=cols[i], label='{}'.format(trials[i]), linestyle='None'))

        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.set_xlabel('$r^{us-only}_{rnn}$ (spikes/s)')
        ax.set_ylabel('$r^{cs-only}_{rnn}$ (spikes/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -5))
        ax.spines['bottom'].set_position(('data', -5))
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.legend(handles=legend_handles, title='Trial #',frameon=False,ncol=1,bbox_to_anchor=(1, 1),
                  title_fontsize=SMALL_SIZE)
        
        #plt.savefig('Sub_his.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
        
        trials = net.n_trial*np.linspace(0,1,int(100/params['every_perc']))
        fig, ax = plt.subplots(figsize=(2,1.5))
        ax.plot(trials,net.E,linewidth=.5,alpha=.3)
        ax.plot(trials,np.average(net.E,axis=1),c='green',linewidth=1.5)
        ax.axhline(y=1,c='gray',linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.05*net.n_trial))
        ax.spines['bottom'].set_position(('data', -.05))
        ax.xaxis.set_major_locator(MultipleLocator(.5*net.n_trial))
        ax.xaxis.set_minor_locator(MultipleLocator(.25*net.n_trial))
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_xlim([0,net.n_trial])
        ax.set_ylabel('Expectation $E$')
        ax.set_xlabel('Trials')
        
        color = 'red'
        ax1 = ax.twinx()
        #ax1.plot(trials,np.var(US_est_hist-US,axis=2),color=color,linewidth=.5,alpha=.5)
        ax1.plot(trials,np.var(US_est_hist-US,axis=(1,2)),color=color,linewidth=1.5)
        ax1.set_ylabel('Var[$r_{US}$,$\hat{r}_{US}$]', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['right'].set_position(('data', 1.05*net.n_trial))
        ax1.yaxis.set_major_locator(MultipleLocator(.1))
        ax1.set_ylim([0,.25])
        
        #plt.savefig('Cond_his.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
        plt.show()
    
        # Scatterplot of actual and decoded US digits
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0, linewidth=1)
        ax.scatter(US.flatten()+np.random.normal(scale=.02,size=np.size(US)),
                   US_est_hist[tr,:].flatten(),s=.25,color='green',alpha=.5,zorder=1)
        ax.set_xlim([-.2,1.2])
        ax.set_ylim([-.2,1.2])
        ax.set_xlabel('$r_{US}$ element')
        ax.set_ylabel('$\hat{r}_{US}$ element')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.25))
        ax.spines['bottom'].set_position(('data', -.25))
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.xaxis.set_minor_locator(MultipleLocator(.25))
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        
        #plt.savefig('USdec.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
        plt.show()
        
        # Plot of CS, US, and jointly induced responses as a function of trial number
        
        fr = [net.Phi_est,net.Phi,net.Phi_est_US]
        snaps = np.array([0,2,49])
        trials = (snaps+1)/100 * net.n_trial; trials = trials.astype('int')
        
        labels = ['CS only', 'US only', 'CS+US']
        
        fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)

        for i, snap in enumerate(snaps):
            for j, Phi in enumerate(fr):
                ax = axes[j, i]
                if j==1:
                    im = ax.imshow(1000*Phi.T, cmap='viridis', aspect='auto')
                else:
                    im = ax.imshow(1000*Phi[snap].T, cmap='viridis', aspect='auto')
                if j==0:
                    ax.set_title('Trial {}'.format(trials[i]))
                #ax.set_xticks([0, 8, 16])
                #ax.set_yticks([0, 32, 64])
                if i == 0:
                    ax.set_ylabel(labels[j], rotation=90, ha='right', va='center')
                    ax.yaxis.set_label_coords(-0.25, 0.6)  # Set the coordinates of the label
                
        axes[2, 0].set_xlabel('CS-US pair')
        axes[0, 2].yaxis.set_label_position('right')
        axes[0, 2].set_ylabel('RNN unit')

        # Add a shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.08, 0.03, 0.5])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Firing rate (spikes/sec)')

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        
        #plt.savefig('firing_rates.png',bbox_inches='tight',format='png',dpi=300)
        plt.show()
        

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

elif n_CS == 2:
    
    trials = net.n_trial*np.linspace(0,1,int(100/params['every_perc']))
    E_1 = net.E_1  # Shape: (n_trials, n_pat)
    E_2 = net.E_2  # Shape: (n_trials, n_pat)
    E = E_1 + E_2
    E_max = np.max(E); R_max = np.ceil(10*np.max([E_max,1]))/10
    
    fig, ax = plt.subplots(figsize=(1.5,1.5))
    
    # Plot individual pattern traces (like the multiple associations style)
    for i in range(net.n_pat):
        ax.plot(trials,E_1[:,i], c='dodgerblue', alpha=0.3, linewidth=0.5)
        ax.plot(trials,E_2[:,i], c='darkorange', alpha=0.3, linewidth=0.5)
        ax.plot(trials,E[:,i], c='green', alpha=0.3, linewidth=0.5)
    
    # Plot mean lines (like network2 style)
    ax.plot(trials,np.mean(E_1, axis=1), c='dodgerblue', linewidth=2, zorder=1)
    ax.plot(trials,np.mean(E_2, axis=1), c='darkorange', linewidth=2, zorder=1)
    ax.plot(trials,np.mean(E, axis=1), c='green', linewidth=2, zorder=1)
    
    # Standard network2 formatting
    ax.axhline(y=1,c='gray',linestyle='-',linewidth=.5)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Expectation $E$')
    ax.set_xlim([0,trials[-1]])
    ax.set_ylim([-.02*R_max,1.02*R_max])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', -.05*trials[-1]))
    ax.spines['bottom'].set_position(('data', -.05*R_max))
    ax.xaxis.set_major_locator(MultipleLocator(int(trials[-1]/2)))
    ax.xaxis.set_minor_locator(MultipleLocator(int(trials[-1]/4)))
    ax.yaxis.set_major_locator(MultipleLocator(R_max/2))
    ax.yaxis.set_minor_locator(MultipleLocator(R_max/4))
    
    #plt.savefig('Cond_his_mult.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
    plt.show()

# Conditioning plot

if net.est_every:
    n_trial = net.n_trial
    
    if n_CS == 1:
        E = net.E
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(E,label='$E$',c='green',linewidth=2)
        ax.axhline(y=1,c='gray',linestyle='-',linewidth=.5)
        #ax.axvline(x=10,linestyle='dotted',c='darkorange',linewidth=1.5,label='Extinction',zorder=0)
        ax.set_xlabel('Trials')
        ax.set_ylabel('Expectation $E$')
        ax.set_xlim([-.02*n_trial,n_trial])
        ax.set_ylim([-.02,1.02])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -.05*n_trial))
        ax.spines['bottom'].set_position(('data', -.05))
        ax.xaxis.set_major_locator(MultipleLocator(int(n_trial/2)))
        ax.xaxis.set_minor_locator(MultipleLocator(n_trial/4))
        ax.yaxis.set_major_locator(MultipleLocator(1/2))
        ax.yaxis.set_minor_locator(MultipleLocator(1/4))
        #fig.legend(frameon=False,loc='right')
        
    elif n_CS == 2:
        E_1 = net.E_1
        E_2 = net.E_2
        E = E_1 + E_2
        E_max = np.max(E); R_max = np.ceil(10*np.max([E_max,1]))/10
        
        fig, ax = plt.subplots(figsize=(1.5,1.5))
        ax.plot(E_1,c='dodgerblue',linewidth=2,zorder=1)
        ax.plot(E_2,c='darkorange',linewidth=2,zorder=1)
        #ax.plot([],[],linestyle='',label='\n')
        #ax.axvline(x=100,linestyle='dotted',c='darkorange',linewidth=1.5,label='$CS_2$ presented, $CS_1$ removed',zorder=0)
        #ax.axvline(x=200,linestyle='dotted',c='green',linewidth=1.5,label='Both $CS$s presented',zorder=0)
        ax.plot(E,c='green',linewidth=2,zorder=1)
        ax.axhline(y=1,c='gray',linestyle='-',linewidth=.5)
        ax.set_xlabel('Trials')
        ax.set_ylabel('Expectation $E$')
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
        #fig.legend(frameon=False,ncol=1,bbox_to_anchor=(.85, 1.15))
        
    
    #plt.savefig('exp.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
    #plt.savefig('exp.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
    plt.show()

if n_CS == 1 and net.trial_dyn:
    
    trials = [0,2,14]
    t = np.linspace(0,net.t_dur,int(net.t_dur/net.dt))
    
    dW_rec = net.dW_rec.reshape(*net.dW_rec.shape[:1], -1, *net.dW_rec.shape[-1:])
    dW_fb = net.dW_fb.reshape(*net.dW_fb.shape[:1], -1, *net.dW_fb.shape[-1:])
    
    fig, axs = plt.subplots(5, 3, figsize=(6,6), sharex = True, sharey = 'row')
    
    for j, trial in enumerate(trials):
        
        # Expectation
        axs[0,j].plot(t,net.E_tr[trial,:])
        axs[0,j].set_title('Trial {}'.format(trial+1))
        axs[0,j].set_ylabel('$E(t-t_{syn})$')
        
        # Neuromodulator concentration
        axs[1,j].plot(t,net.eta[trial,:])
        axs[1,j].set_ylabel('Learning rate $\eta$')
        
        # Firing rate error
        axs[2,j].plot(t,net.error[trial,:].T*1000,alpha=.1)
        axs[2,j].set_ylabel('Firing rate error \n $f(V^s_i)-f(p^\prime \, V^d_i)$')
        
        # PSPs
        axs[3,j].plot(t,net.PSP[trial,:].T,alpha=.1)
        axs[3,j].set_ylabel('Post-synaptic \n potential $P_j$')
        
        # Weight change
        axs[4,j].plot(t,dW_rec[trial,:].T,alpha=.1)
        axs[4,j].plot(t,dW_fb[trial,:].T,alpha=.1)
        axs[4,j].set_ylabel('$\Delta W$')
        
        if j == 0:
            axs[4,j].set_xlabel('Time (s)')
        else:
            axs[4,j].set_xlabel('')
        
    for ax in axs.flat:
        ax.label_outer()
        
    fig.tight_layout()

    #plt.savefig('trial_dyn.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
    plt.show()

''' 
# Similarity curves
d = np.linspace(-3,3,100)
n2 = np.exp(-d**2); n4 = np.exp(-d**4); n6 = np.exp(-d**6)

fig, ax = plt.subplots(figsize=(2,1.5))
ax.plot(d,n2,label='Gaussian (order=2)')
ax.plot(d,n4,label='Hyper-gaussian (order=4)')
ax.plot(d,n6,label='Hyper-gaussian (order=6)')
ax.set_xlabel('Distance from $r^{US}$')
ax.set_ylabel('Surprise')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.legend(frameon=False,loc='right',bbox_to_anchor=(1.1, 1.3))

#plt.savefig('rbf_kernels.png',bbox_inches='tight',format='png',dpi=300)
'''