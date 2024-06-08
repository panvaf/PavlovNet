"""
Generate plots from multiple networks.
"""

import os
from pathlib import Path
import pickle
import util
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit

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
    'a': 1,              # deviation from self-consistency
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
    'R': None,           # reward associated with every US
    'S': None,           # sign of neurons
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'mem_net_id': 'MemNet64tdur3iter1e5Noise0.1',  # Memory RNN to load
    'out': True,         # whether to feed output of RNN to associative net
    'est_every': False,  # whether to estimate US and reward after every trial
    'DA_plot': False,    # whether to keep track of expected reward within trial
    'trial_dyn': False,  # whether to store trial dynamics
    'GiveR': True,       # whether to provide reward upon US presentation
    'flip': False,       # whether to flip the US-CS associations mid-learning
    'extinct': False,    # whether to undergo extinction of learned association
    'reacquire': False,  # whether to undergo extinction and reacquisition of learned association
    'exact': False,      # whether to demand an exact Hamming distance between patterns
    'low': 1,           # lowest possible reward
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'T': 1,              # temporal window for averaging firing rates for BCM rule
    'run': 0,            # number of run for many runs of same simulation
    'm': 2               # order of gaussian for radial basis function
    }

data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'

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


# ISI curve

delays = np.arange(-1,9)

perc_CR = np.zeros((len(delays),params['n_pat']))
perc_CR_mean = np.zeros(len(delays))
perc_CR_025 = np.zeros(len(delays))
perc_CR_975 = np.zeros(len(delays))

for i, t_d in enumerate(delays):
    
    params['US_ap'] = params['CS_disap'] + t_d
    params['t_dur'] = params['US_ap'] + 1
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DAreprod'
    with open(data_path+filename+'.pkl', 'rb') as f:
        net = pickle.load(f)
    
    # Find mean and 95 % intervals of conditioned response across CSs
    perc_CR[i] = np.divide(net.R_est,net.R)
    perc_CR_mean[i] = np.mean(perc_CR[i])
    (perc_CR_025[i], perc_CR_975[i]) = st.t.interval(alpha=0.95,
            df=len(perc_CR[i])-1, loc=np.mean(perc_CR[i]), scale=st.sem(perc_CR[i]))

params['CS_disap'] = 2; params['US_ap'] = 1; params['t_dur'] = 2

fig, ax = plt.subplots(figsize=(2,1.5))
plt.scatter(delays,perc_CR_mean,color = 'green',s=10)
plt.errorbar(delays,perc_CR_mean,[perc_CR_mean-perc_CR_025,perc_CR_975-perc_CR_mean],color = 'green',linestyle='')
plt.ylabel('Expectation')
plt.xlabel('$t_d$ (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -1.5))
ax.spines['bottom'].set_position(('data', -.05))
plt.xlim([delays[0]-.3,delays[-1]+.3])
plt.ylim([0,1])
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))

#plt.savefig('trace_cond.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('trace_cond.eps',bbox_inches='tight',format='eps',dpi=300)

# Rate of acquisition as a function of reward size plot

params['n_trial'] = 1e2; params['n_pat'] = 1
params['eta'] = 5e-4; params['est_every'] = True

rewards = [1,.7,.4]
base_color = 'red'
colors = [util.saturation_mult(base_color,1.2),util.saturation_mult(base_color,.6),util.saturation_mult(base_color,.3)]
# alternative : 1.5, 0.7, 0.2

fig, ax = plt.subplots(figsize=(1.5,1.5))

for i, reward in enumerate(rewards):
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20DAreprodR' + str(reward).replace('.','')
    with open(data_path+filename+'.pkl', 'rb') as f:
        net = pickle.load(f)
    
    norm_R = net.R_est/net.R
    n_trial = net.n_trial
    
    ax.plot(norm_R,label='$R=$'+'${}$'.format(reward),c=colors[i],linewidth=2)
    
ax.axhline(y=1,c='black',linestyle='--',linewidth=2)
ax.set_xlabel('Trials')
ax.set_ylabel('Normalised Reward')
ax.set_xlim([0,n_trial])
ax.set_ylim([-.02,1.02])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.05*n_trial))
ax.spines['bottom'].set_position(('data', -.05))
ax.xaxis.set_major_locator(MultipleLocator(int(n_trial/2)))
ax.xaxis.set_minor_locator(MultipleLocator(int(n_trial/4)))
ax.yaxis.set_major_locator(MultipleLocator(1/2))
ax.yaxis.set_minor_locator(MultipleLocator(1/4))
fig.legend(frameon=False,loc='right',bbox_to_anchor=(1.75,.75))

# Rate of acquisition as a function of number of entrained patterns

n_pat = [2,4,8,16]
n_trial = [100,100,200,2000]
runs = [0]
params['eta'] = 5e-3
params['dale'] = False; params['filter'] = True; params['mem_net_id'] = None; params['out'] = False 

t_learned = np.zeros((len(n_pat),len(runs)))
t_learned_mean = np.zeros(len(n_pat))
t_learned_025 = np.zeros(len(n_pat))
t_learned_975 = np.zeros(len(n_pat))

fig, axs = plt.subplots(2,2,sharex=False,sharey=True,figsize=(5,4))

for i, pat in enumerate(n_pat):
    
    params['n_pat'] = pat
    params['n_trial'] = n_trial[i]
    
    for j, run in enumerate(runs):
        
        params['run'] = run
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20DA' + 'reprod'
        with open(data_path+filename+'.pkl', 'rb') as f:
            net = pickle.load(f)
        
        R_est = net.R_est
        loc = np.where(np.all(R_est>.75,1))[0][0]
        
        if j==0:
            
            k = i // 2
            l = i % 2
            
            axs[k,l].plot(R_est,linewidth=.5,alpha=.5)
            axs[k,l].plot(np.average(R_est,axis=1),c='green',linewidth=1)
            axs[k,l].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
            axs[k,l].axvline(x=loc,c='red',ls='--',linewidth=1)
            axs[k,l].set_title('$N^{as}=$'+'${}$'.format(pat))
            axs[k,l].set_xlim([0,n_trial[i]])
            axs[k,l].spines['top'].set_visible(False)
            axs[k,l].spines['right'].set_visible(False)
            axs[k,l].spines['left'].set_position(('data', -.05*n_trial[i]))
            axs[k,l].spines['bottom'].set_position(('data', -.05))
            axs[k,l].xaxis.set_major_locator(MultipleLocator(int(n_trial[i]/2)))
            axs[k,l].xaxis.set_minor_locator(MultipleLocator(int(n_trial[i]/4)))
            
            if i==2:
                axs[k,l].set_ylabel('Reward')
                axs[k,l].set_xlabel('Trials')
                axs[k,l].set_ylim([-.02,1.02])
                axs[k,l].yaxis.set_major_locator(MultipleLocator(1/2))
                axs[k,l].yaxis.set_minor_locator(MultipleLocator(1/4))
                
        t_learned[i,j] = loc
    
    # Find mean and 95 % intervals of time to learn across runs
    t_learned_mean[i] = np.mean(t_learned[i])
    (t_learned_025[i], t_learned_975[i]) = st.t.interval(alpha=0.95,
            df=len(t_learned[i])-1, loc=np.mean(t_learned[i]), scale=st.sem(t_learned[i]))
    
plt.tight_layout()
#plt.savefig('1a.png',bbox_inches='tight',format='png',dpi=300)

coeff = np.polyfit(n_pat,np.log(t_learned_mean),1)
x = np.linspace(n_pat[0],n_pat[-1],1000)
fit_ln = np.exp(coeff[0]*x+coeff[1])

p0 = (coeff[0], coeff[1])
[m, t], _ = curve_fit(util.Exp,n_pat,t_learned_mean,p0)
fit = np.exp(m*x+t)

fig, ax = plt.subplots(figsize=(2,1.5))
plt.loglog(x,fit_ln,color='black',linewidth=1,label='Exponential Fit')
ax.set_xscale("log", base=2)
plt.scatter(n_pat,t_learned_mean,color = 'green',s=10,label='Data')
plt.errorbar(n_pat,t_learned_mean,[t_learned_mean-t_learned_025,t_learned_975-t_learned_mean],color = 'green',linestyle='')
plt.ylabel('# trials to learn')
plt.xlabel('$N^{as}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_position(('data', tau_tot[0]-15))
#ax.spines['bottom'].set_position(('data', 240))
plt.legend(loc='upper center',markerscale=1,frameon=False)


# Effect of Hamming distance on learning rate

H_d = [4,6,8,10]
n_trial = [300,200,200,200]
runs = [0]
params['exact'] = True
params['n_pat'] = 8

t_learned = np.zeros((len(H_d),len(runs)))
t_learned_mean = np.zeros(len(H_d))
t_learned_025 = np.zeros(len(H_d))
t_learned_975 = np.zeros(len(H_d))

fig, axs = plt.subplots(2,2,sharex=False,sharey=True,figsize=(5,4))

for i, h_d in enumerate(H_d):
    
    params['H_d'] = h_d
    params['n_trial'] = n_trial[i]
    
    for j, run in enumerate(runs):
        
        params['run'] = run
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20DA'
        with open(data_path+filename+'.pkl', 'rb') as f:
            net = pickle.load(f)
        
        R_est = net.R_est
        loc = np.where(np.all(R_est>.75,1))[0][0]
        
        if j==0:
            
            k = i // 2
            l = i % 2
            
            axs[k,l].plot(R_est,linewidth=.5,alpha=.5)
            axs[k,l].axhline(y=1,c='black',linestyle='--',linewidth=1.5)
            axs[k,l].plot(np.average(R_est,axis=1),c='green',linewidth=1)
            axs[k,l].axvline(x=loc,c='red',ls='--',linewidth=1)
            axs[k,l].set_title('$H^{d}=$'+'${}$'.format(h_d))
            axs[k,l].set_xlim([0,n_trial[i]])
            axs[k,l].spines['top'].set_visible(False)
            axs[k,l].spines['right'].set_visible(False)
            axs[k,l].spines['left'].set_position(('data', -.05*n_trial[i]))
            axs[k,l].spines['bottom'].set_position(('data', -.05))
            axs[k,l].xaxis.set_major_locator(MultipleLocator(int(n_trial[i]/2)))
            axs[k,l].xaxis.set_minor_locator(MultipleLocator(int(n_trial[i]/4)))
            
            if i==2:
                axs[k,l].set_ylabel('Reward')
                axs[k,l].set_xlabel('Trials')
                axs[k,l].set_ylim([-.02,1.02])
                axs[k,l].yaxis.set_major_locator(MultipleLocator(1/2))
                axs[k,l].yaxis.set_minor_locator(MultipleLocator(1/4))
                
        t_learned[i,j] = loc
    
    # Find mean and 95 % intervals of time to learn across runs
    t_learned_mean[i] = np.mean(t_learned[i])
    (t_learned_025[i], t_learned_975[i]) = st.t.interval(alpha=0.95,
            df=len(t_learned[i])-1, loc=np.mean(t_learned[i]), scale=st.sem(t_learned[i]))
    
plt.tight_layout()
#plt.savefig('1b.png',bbox_inches='tight',format='png',dpi=300)

coeff_inv,_,_,_ = np.linalg.lstsq(1/np.array(H_d)[:,np.newaxis],t_learned_mean)
x = np.linspace(H_d[0],H_d[-1],1000)
fit_inv = coeff_inv/x

fig, ax = plt.subplots(figsize=(2,1.5))
plt.plot(x,fit_inv,color='black',linewidth=1,label='Fit')
plt.scatter(H_d,t_learned_mean,color = 'green',s=10,label='Data')
plt.errorbar(H_d,t_learned_mean,[t_learned_mean-t_learned_025,t_learned_975-t_learned_mean],color = 'green',linestyle='')
plt.ylabel('# trials to learn')
plt.xlabel('$H^{d}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_position(('data', tau_tot[0]-15))
#ax.spines['bottom'].set_position(('data', 240))
plt.legend(loc='upper center',markerscale=1,frameon=False)

#plt.savefig('4c.eps',bbox_inches='tight',format='eps',dpi=300)

params['n_pat'] = 16; params['H_d'] = 8; params['n_trial'] = 1e3
params['dale'] = True; params['out'] = True; params['exact'] = False
params['est_every'] = False; params['filter'] = False
params['mem_net_id'] = 'MemNet64tdur3iter1e5Noise0.1'


# BCM for different averaging windows and trial configurations

params['eta'] = 0.3
params['rule'] = 'BCM'
Ts = [0.1,0.2,0.3,0.4]
tUSs = [0.5, 0.75, 1]
colors = ['purple','red','green']

exp_mean = np.zeros((len(Ts),len(tUSs)))
exp_025 = np.zeros((len(Ts),len(tUSs)))
exp_975 = np.zeros((len(Ts),len(tUSs)))

for j, tUS in enumerate(tUSs):
    
    params['US_ap'] = tUS
    params['CS_disap'] = tUS + 1
    params['t_dur'] = tUS + 1
    
    for i, T in enumerate(Ts):
        
        params['T'] = T
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20reprod'
        with open(data_path+filename+'.pkl', 'rb') as f:
            net = pickle.load(f)
            
        exp = net.R_est[-1]
        exp_mean[i,j] = np.mean(exp)
        (exp_025[i,j], exp_975[i,j]) = st.t.interval(alpha=0.95,
                df=len(exp)-1, loc=np.mean(exp), scale=st.sem(exp))

fig, ax = plt.subplots(figsize=(2,1.5))
for j in range(len(tUSs)):
    plt.scatter(Ts,exp_mean[:,j],color = colors[j],s=10,label='{}'.format(tUSs[j]))
    plt.errorbar(Ts,exp_mean[:,j],[exp_mean[:,j]-exp_025[:,j],exp_975[:,j]-exp_mean[:,j]],color=colors[j],linestyle='')
plt.ylabel('Expectation')
fig.gca().set_xlabel(r'$\tau_\theta$ (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0.07))
ax.spines['bottom'].set_position(('data', -.05))
#plt.xlim([Ts[0],Ts[-1]])
plt.ylim([0,1])
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))   
fig.legend(frameon=False,loc='right',bbox_to_anchor=(.45,.8), title="$t_{US}$ (s)")

#plt.savefig('BCM_sweep.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('BCM_sweep.eps',bbox_inches='tight',format='eps',dpi=300)