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
from scipy.stats import sem
import matplotlib.lines as mlines

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
    'extinct': False,    # whether to undergo extinction of learned association
    't_wait': 5,         # time after US_ap that its considered an extinction trial
    'reacquire': False,  # whether to undergo extinction and reacquisition of learned association
    'exact': False,      # whether to demand an exact Hamming distance between patterns
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'T': 1,              # temporal window for averaging firing rates for BCM rule
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
    'a': 0.95,           # deviation from self-consistency
    'n_trial': 1e2,      # number of trials
    't_dur': 2,          # duration of trial
    'CS_2_ap_tr': 0,     # trial number in which CS 2 appears
    'US_ap': 1,          # time in trial that US appears
    'train': True,       # whether to train network or not
    'fun': 'logistic',   # activation function of associative network
    'every_perc': 1,     # store errors this often
    'dale': True,        # whether the network respects Dale's law
    'I_inh': 0,          # global inhibition to dendritic compartment
    'est_every': True,   # whether to estimate US and expectation after every trial
    'overexp': False,    # whether to test for overexpectation effects
    'salience': 1,       # relative saliance of CSs
    'cont': [1,1],       # contingencies of CSs
    'cond_dep': False,   # whether one CS is conditionally dependent on the other
    'filter': False,     # whether to filter the learning dynamics
    'rule': 'Pred',      # learning rule used in associative network
    'norm': None,        # normalization strenght for learning rule
    'm': 2               # order of gaussian for radial basis function
    }

data_path = os.path.join(str(Path(os.getcwd()).parent),'trained_networks')

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


# Multiple conditioning examples

runs = np.arange(5)

fig, ax = plt.subplots(figsize=(2, 1.5))

for i, run in enumerate(runs):
    
    params['run'] = run
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'

    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)

    trials = net.n_trial * np.linspace(0, 1, int(100 / params['every_perc']))
    avg_E = np.mean(net.E, axis=1)
    sem_E = sem(net.E, axis=1)
    ci_lower = avg_E - 1.96 * sem_E
    ci_upper = avg_E + 1.96 * sem_E

    ax.plot(trials, avg_E, linewidth=1.5)
    ax.fill_between(trials, ci_lower, ci_upper, alpha=0.3)
    
ax.axhline(y=1, c='gray', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.05 * net.n_trial))
ax.spines['bottom'].set_position(('data', -.05))
ax.xaxis.set_major_locator(MultipleLocator(.5 * net.n_trial))
ax.xaxis.set_minor_locator(MultipleLocator(.25 * net.n_trial))
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))
ax.set_xlim([0, net.n_trial])
ax.set_ylabel('Expectation $E$')
ax.set_xlabel('Trials')

#plt.savefig('Exp_multiple.png',bbox_inches='tight',format='png',dpi=300,transparent=True)


# Do the same for errors

fig, ax = plt.subplots(figsize=(2, 1.5))

for i, run in enumerate(runs):
    
    params['run'] = run
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'

    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)
        
    US = net.US
    US_est_hist = net.US_est
        
    ax.plot(trials,np.var(US_est_hist-US,axis=(1,2)),linewidth=1.5)

ax.set_ylabel('Var[$r_{US}$,$\hat{r}_{US}$]')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(('data', -.05 * net.n_trial))
ax.spines['bottom'].set_position(('data', -.01))
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.set_ylim([0,.25])
ax.xaxis.set_major_locator(MultipleLocator(.5 * net.n_trial))
ax.xaxis.set_minor_locator(MultipleLocator(.25 * net.n_trial))
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.yaxis.set_minor_locator(MultipleLocator(.05))
ax.set_xlabel('Trials')

#plt.savefig('var_multiple.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

params['run'] = 0


# Demonstrate insensitivity to trial details

tUSs = [0.5,0.75,1]

exp_mean = np.zeros(len(tUSs))
exp_025 = np.zeros(len(tUSs))
exp_975 = np.zeros(len(tUSs))

for i, tUS in enumerate(tUSs):
    
    params['US_ap'] = tUS
    params['CS_disap'] = tUS + 1
    params['t_dur'] = tUS + 1
    
        
    filename = util.filename(params) + 'gsh3gD2gL1taul20'
    
    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)
        
    exp = net.E[-1]
    exp_mean[i] = np.mean(exp)
    (exp_025[i], exp_975[i]) = st.t.interval(confidence=0.95,
            df=len(exp)-1, loc=np.mean(exp), scale=st.sem(exp))

fig, ax = plt.subplots(figsize=(1.2,1.5))
plt.scatter(tUSs,exp_mean,color = 'green',s=10)
plt.errorbar(tUSs,exp_mean,[exp_mean-exp_025,exp_975-exp_mean],color='green',linestyle='')
plt.ylabel('Expectation $E$')
plt.title('Predictive learning')
fig.gca().set_xlabel(r'$t_{us-on}$ (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0.45))
ax.spines['bottom'].set_position(('data', -.05))
#plt.xlim([Ts[0],Ts[-1]])
plt.ylim([0,1])
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))
ax.xaxis.set_major_locator(MultipleLocator(.25))

#plt.savefig('tUSon_sweep.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('tUSon_sweep.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)


# Contingency and speed of learning plot

conts = [0.8,0.6,0.4]
colors = ['dodgerblue','firebrick','magenta']

fig, ax = plt.subplots(figsize=(2, 1.5))

for i, cont in enumerate(conts):
    
    params2['cont'] = [cont,0]
    
    filename = util.filename2(params2) + 'gsh3gD2gL1taul20'

    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)
        
    E_1 = net.E_1
    n_trial = net.n_trial
        
    ax.plot(E_1,linewidth=2,label=cont,color=colors[i])
    ax.set_ylabel('Var[$r_{US}$,$\hat{r}_{US}$]')

ax.axhline(y=1,c='gray',linestyle='-',linewidth=.5)
ax.set_xlabel('Trials')
ax.set_ylabel('Expectation $E$')
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
fig.legend(frameon=False,ncol=1,bbox_to_anchor=(1, .6),title='$P(CS)$')

#plt.savefig('cont_speed.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('cont_speed.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)

params2['cont'] = [1,1]


# ISI curve

delays = np.arange(-1,9)

perc_CR = np.zeros((len(delays),params['n_pat']))
perc_CR_mean = np.zeros(len(delays))
perc_CR_025 = np.zeros(len(delays))
perc_CR_975 = np.zeros(len(delays))

for i, t_d in enumerate(delays):
    
    # Trace conditioning examples have shorter CS duration
    if i>0:   
        params['US_ap'] = 1 + t_d
        params['t_dur'] = 2 + t_d
        params['CS_disap'] = 1
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'
    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)
    
    # Find mean and 95 % intervals of conditioned response across CSs
    perc_CR[i] = net.E[-1]
    perc_CR_mean[i] = np.mean(perc_CR[i])
    (perc_CR_025[i], perc_CR_975[i]) = st.t.interval(confidence=0.95, 
        df=len(perc_CR[i])-1, loc=np.mean(perc_CR[i]), scale=st.sem(perc_CR[i]))

fig, ax = plt.subplots(figsize=(2,1.5))
plt.scatter(delays,perc_CR_mean,color = 'green',s=10)
plt.errorbar(delays,perc_CR_mean,[perc_CR_mean-perc_CR_025,perc_CR_975-perc_CR_mean],color = 'green',linestyle='')
plt.ylabel('Expectation $E$')
plt.xlabel('$t_{delay}$ (s)')
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

#plt.savefig('trace_cond.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('trace_cond.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)

params['CS_disap'] = 2; params['US_ap'] = 1; params['t_dur'] = 2

# Stimulus substitution and conditioning history plots

delays = np.arange(-1,6,2)
colors = ['green','firebrick','dodgerblue','magenta']

US_est = np.zeros((len(delays),params['n_pat'],params['n_in']))
US = np.zeros(US_est.shape)

fig, ax = plt.subplots(figsize=(2, 1.5))

for i, t_d in enumerate(delays):
    
    if i>0:
        params['US_ap'] = 1 + t_d
        params['t_dur'] = 2 + t_d
        params['CS_disap'] = 1
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'

    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)

    trials = net.n_trial * np.linspace(0, 1, int(100 / params['every_perc']))
    avg_E = np.mean(net.E, axis=1)
    sem_E = sem(net.E, axis=1)
    ci_lower = avg_E - 1.96 * sem_E
    ci_upper = avg_E + 1.96 * sem_E

    ax.plot(trials, avg_E, linewidth=1.5, color = colors[i])
    ax.fill_between(trials, ci_lower, ci_upper, alpha=0.3, color = colors[i])
    
    # Store for future use
    US_est[i] = net.US_est[49]
    US[i] = net.US
    
ax.axhline(y=1, c='gray', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.05 * net.n_trial))
ax.spines['bottom'].set_position(('data', -.05))
ax.xaxis.set_major_locator(MultipleLocator(.5 * net.n_trial))
ax.xaxis.set_minor_locator(MultipleLocator(.25 * net.n_trial))
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))
ax.set_xlim([0, net.n_trial])
ax.set_ylabel('Expectation $E$')
ax.set_xlabel('Trials')

#plt.savefig('trace_multiple.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

x_bias = np.linspace(-.15,.15,4)

fig, ax = plt.subplots(figsize=(1.5,1.5))
ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0, linewidth=1)

legend_handles = []
for i, t_d in enumerate(delays):
    ax.scatter(US[i].flatten()+x_bias[i]+np.random.normal(scale=.02,size=np.size(US[i])),
               US_est[i].flatten(),s=.25,color=colors[i],alpha=.5,zorder=1)
    # Create custom legend handles
    legend_handles.append(mlines.Line2D([], [], marker='o', markersize=1.5, 
            color=colors[i], label=t_d, linestyle='None'))
    
ax.set_xlim([-.5,1.5])
ax.set_ylim([-.5,1.5])
ax.set_xlabel('$r_{US}$ element')
ax.set_ylabel('$\hat{r}_{US}$ element')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.55))
ax.spines['bottom'].set_position(('data', -.55))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(.5))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(.5))
fig.legend(handles=legend_handles, title='$t_{delay}$ (s)',frameon=False,ncol=1,bbox_to_anchor=(1.3, .8),
          title_fontsize=SMALL_SIZE)

#plt.savefig('trace_multiple_sub.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

params['CS_disap'] = 2; params['US_ap'] = 1; params['t_dur'] = 2


# Reacquisition with and without US flip

params['eta'] = 5e-4; params['n_trial'] = 5e2; params['n_pat'] = 1
params['est_every'] = True; params['reacquire'] = True; params['t_dur'] = 7

flips = [False,True]
labels = ['Same $US$','Different $US$']
colors = ['dodgerblue','red']

fig, ax = plt.subplots(figsize=(3.5,1.5))

for i, flip in enumerate(flips):
    
    params['flip'] = flip
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'

    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)

    n_trial = net.n_trial
    ax.plot(net.E,label=labels[i],linewidth=2,color=colors[i],zorder=1-i)
    
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
ax.xaxis.set_major_locator(MultipleLocator(int(n_trial/4)))
ax.xaxis.set_minor_locator(MultipleLocator(n_trial/8))
ax.yaxis.set_major_locator(MultipleLocator(1/2))
ax.yaxis.set_minor_locator(MultipleLocator(1/4))
fig.legend(frameon=False,ncol=1,bbox_to_anchor=(.75, .7))

#plt.savefig('reacquisition.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('reacquisition.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)

params['est_every'] = False; params['reacquire'] = False; params['flip'] = False
params['t_dur'] = 2

# Rate of acquisition as a function of number of entrained patterns

n_pat = [2,4,8,16]
n_trial = [100,100,200,1000]
runs = np.arange(5)
params['eta'] = 5e-3

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
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20'
        
        with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
            net = pickle.load(f)
        
        E = net.E
        trials = net.n_trial*np.linspace(0,1,int(100/net.every_perc))
        
        loc = np.where(np.average(E,axis=1)>.8)[0][0] * net.n_trial/100
        
        if j==0:
            
            k = i // 2
            l = i % 2
            
            axs[k,l].plot(trials,E,linewidth=.5,alpha=.5)
            axs[k,l].plot(trials,np.average(E,axis=1),c='green',linewidth=1)
            axs[k,l].axhline(y=1,c='gray',linestyle='-',linewidth=.5)
            axs[k,l].axvline(x=loc,c='red',ls='-',linewidth=1)
            axs[k,l].set_title('$N_{stim}=$'+'${}$'.format(pat))
            axs[k,l].set_xlim([0,n_trial[i]])
            axs[k,l].spines['top'].set_visible(False)
            axs[k,l].spines['right'].set_visible(False)
            axs[k,l].spines['left'].set_position(('data', -.05*n_trial[i]))
            axs[k,l].spines['bottom'].set_position(('data', -.05))
            axs[k,l].xaxis.set_major_locator(MultipleLocator(int(n_trial[i]/2)))
            axs[k,l].xaxis.set_minor_locator(MultipleLocator(int(n_trial[i]/4)))
            
            if i==0:
                axs[k,l].set_ylabel('Expectation $E$')
                axs[k,l].set_ylim([-.02,1.02])
                axs[k,l].yaxis.set_major_locator(MultipleLocator(1/2))
                axs[k,l].yaxis.set_minor_locator(MultipleLocator(1/4))
                
            if i==2:
                axs[k,l].set_xlabel('Trials')
                
        t_learned[i,j] = loc
    
    # Find mean and 95 % intervals of time to learn across runs
    t_learned_mean[i] = np.mean(t_learned[i])
    (t_learned_025[i], t_learned_975[i]) = st.t.interval(confidence=0.95,
            df=len(t_learned[i])-1, loc=np.mean(t_learned[i]), scale=st.sem(t_learned[i]))
    
plt.tight_layout()
#plt.savefig('Npat_examples.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

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
plt.xlabel('$N_{stim}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_position(('data', tau_tot[0]-15))
#ax.spines['bottom'].set_position(('data', 240))
plt.legend(loc='upper center',markerscale=1,frameon=False)

#plt.savefig('Npat_cond.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('Npat_cond.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)


# Effect of Hamming distance on learning rate

H_d = [4,6,8,10]
n_trial = [300,200,200,200]
runs = np.arange(10)
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
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20'
        
        with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
            net = pickle.load(f)
        
        E = net.E
        trials = net.n_trial*np.linspace(0,1,int(100/net.every_perc))
        
        loc = np.where(np.average(E,axis=1)>.8)[0][0] * net.n_trial/100
        
        if j==0:
            
            k = i // 2
            l = i % 2
            
            axs[k,l].plot(trials,E,linewidth=.5,alpha=.5)
            axs[k,l].axhline(y=1,c='gray',linestyle='-',linewidth=.5)
            axs[k,l].plot(trials,np.average(E,axis=1),c='green',linewidth=1)
            axs[k,l].axvline(x=loc,c='red',ls='-',linewidth=1)
            axs[k,l].set_title('$H_d=$'+'${}$'.format(h_d))
            axs[k,l].set_xlim([0,n_trial[i]])
            axs[k,l].spines['top'].set_visible(False)
            axs[k,l].spines['right'].set_visible(False)
            axs[k,l].spines['left'].set_position(('data', -.05*n_trial[i]))
            axs[k,l].spines['bottom'].set_position(('data', -.05))
            axs[k,l].xaxis.set_major_locator(MultipleLocator(int(n_trial[i]/2)))
            axs[k,l].xaxis.set_minor_locator(MultipleLocator(int(n_trial[i]/4)))
            
            if i==0:
                axs[k,l].set_ylabel('Expectation $E$')
                axs[k,l].set_ylim([-.02,1.02])
                axs[k,l].yaxis.set_major_locator(MultipleLocator(1/2))
                axs[k,l].yaxis.set_minor_locator(MultipleLocator(1/4))
                
            if i==2:
                axs[k,l].set_xlabel('Trials')
                
                
        t_learned[i,j] = loc
    
    # Find mean and 95 % intervals of time to learn across runs
    t_learned_mean[i] = np.mean(t_learned[i])
    (t_learned_025[i], t_learned_975[i]) = st.t.interval(confidence=0.95,
            df=len(t_learned[i])-1, loc=np.mean(t_learned[i]), scale=st.sem(t_learned[i]))
    
plt.tight_layout()
#plt.savefig('Hd_examples.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

coeff_inv,_,_,_ = np.linalg.lstsq(1/np.array(H_d)[:,np.newaxis],t_learned_mean)
x = np.linspace(H_d[0],H_d[-1],1000)
fit_inv = coeff_inv/x

fig, ax = plt.subplots(figsize=(2,1.5))
plt.plot(x,fit_inv,color='black',linewidth=1,label='$1/x$ Fit')
plt.scatter(H_d,t_learned_mean,color = 'green',s=10,label='Data')
plt.errorbar(H_d,t_learned_mean,[t_learned_mean-t_learned_025,t_learned_975-t_learned_mean],color = 'green',linestyle='')
plt.ylabel('# trials to learn')
plt.xlabel('$H_{d}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_position(('data', tau_tot[0]-15))
#ax.spines['bottom'].set_position(('data', 240))
plt.legend(loc='upper right',markerscale=1,frameon=False)

#plt.savefig('Hd_cond.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
#plt.savefig('Hd_cond.png',bbox_inches='tight',format='png',dpi=300,transparent=True)

params['n_pat'] = 16; params['H_d'] = 8; params['n_trial'] = 1e3
params['exact'] = False; params['run'] = 0


# Hebbian learning rule for different normalization strengths

params['n_pat'] = 1; params['eta'] = 2e-4; params['rule'] = 'Hebb'
params['est_every'] = True; params['a'] = 1; params['n_trial'] = 1e2

norms = [10,20,40]

fig, ax = plt.subplots(figsize=(1.5, 1.5))
cols = ['dodgerblue','darkorange','green']

legend_handles = []
for i, norm in enumerate(norms):
    
    params['norm'] = norm
    
    filename = util.filename(params) + 'gsh3gD2gL1taul20'

    with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
        net = pickle.load(f)
        
    x = net.Phi.flatten()*1000
    y = net.Phi_est[-1,:].flatten()*1000
    ax.scatter(x, y, s=.25, color=cols[i], alpha=.7, zorder=i+1)

    # Create custom legend handles
    legend_handles.append(mlines.Line2D([], [], marker='o', markersize=1, 
                color=cols[i], label='{}'.format(norm), linestyle='None'))
    
ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black',zorder=0)
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
ax.legend(handles=legend_handles, title='Normalization \n strength',frameon=False,ncol=1,bbox_to_anchor=(1, .8),
          title_fontsize=SMALL_SIZE)

#plt.savefig('Sub_hebb_1patt.png',bbox_inches='tight',format='png',dpi=300,transparent=True)


params['n_pat'] = 16; params['est_every'] = False; params['a'] = 1; params['n_trial'] = 1e3


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
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20'
        
        with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
            net = pickle.load(f)
            
        exp = net.E[-1]
        exp_mean[i,j] = np.mean(exp)
        (exp_025[i,j], exp_975[i,j]) = st.t.interval(confidence=0.95,
                df=len(exp)-1, loc=np.mean(exp), scale=st.sem(exp))

fig, ax = plt.subplots(figsize=(2,1.5))
for j in range(len(tUSs)):
    plt.scatter(Ts,exp_mean[:,j],color = colors[j],s=10,label='{}'.format(tUSs[j]))
    plt.errorbar(Ts,exp_mean[:,j],[exp_mean[:,j]-exp_025[:,j],exp_975[:,j]-exp_mean[:,j]],color=colors[j],linestyle='')
plt.ylabel('Expectation $E$')
plt.title('$a={}$'.format(params['a']))
fig.gca().set_xlabel(r'$\tau_\theta$ (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0.07))
ax.spines['bottom'].set_position(('data', -.05))
#plt.xlim([Ts[0],Ts[-1]])
plt.ylim([0,1])
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))   
fig.legend(frameon=False,loc='right',bbox_to_anchor=(1.3,.6), title="$t_{us-on}$ (s)")

#plt.savefig('BCM_sweep.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('BCM_sweep.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)


# Same for different a

params['a'] = 1.05
tUSs = [0.5,0.75,1]
Ts = [0.2,0.3,0.4,0.5]

exp_mean = np.zeros((len(Ts),len(tUSs)))
exp_025 = np.zeros((len(Ts),len(tUSs)))
exp_975 = np.zeros((len(Ts),len(tUSs)))


for j, tUS in enumerate(tUSs):
    
    params['US_ap'] = tUS
    params['CS_disap'] = tUS + 1
    params['t_dur'] = tUS + 1
    
    for i, T in enumerate(Ts):
        
        params['T'] = T
        
        filename = util.filename(params) + 'gsh3gD2gL1taul20'
        
        with open(os.path.join(data_path,filename+'.pkl'), 'rb') as f:
            net = pickle.load(f)
            
        exp = net.E[-1]
        exp_mean[i,j] = np.mean(exp)
        (exp_025[i,j], exp_975[i,j]) = st.t.interval(confidence=0.95,
                df=len(exp)-1, loc=np.mean(exp), scale=st.sem(exp))

fig, ax = plt.subplots(figsize=(2,1.5))
for j in range(len(tUSs)):
    plt.scatter(Ts,exp_mean[:,j],color = colors[j],s=10,label='{}'.format(tUSs[j]))
    plt.errorbar(Ts,exp_mean[:,j],[exp_mean[:,j]-exp_025[:,j],exp_975[:,j]-exp_mean[:,j]],color=colors[j],linestyle='')
plt.ylabel('Expectation $E$')
plt.title('$a={}$'.format(params['a']))
fig.gca().set_xlabel(r'$\tau_\theta$ (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0.17))
ax.spines['bottom'].set_position(('data', -.05))
#plt.xlim([Ts[0],Ts[-1]])
plt.ylim([0,1])
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))   
fig.legend(frameon=False,loc='right',bbox_to_anchor=(1.3,.6), title="$t_{us-on}$ (s)")

#plt.savefig('BCM_sweep_a105.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('BCM_sweep_a105.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
