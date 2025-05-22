"""
Main classes.
"""

import numpy as np
import util
import assoc_net
from time import time
import os
from pathlib import Path
import torch
import torch.nn as nn
from random import sample, randint
from collections import deque

# File directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'trained_networks')

# Single associative network class (fig. 1b)

class network:
    
    def __init__(self,params,seed=None):
        
        # Random seed
        self.seed = seed
        
        # Set the random seed for NumPy
        if seed is not None:
            np.random.seed(self.seed)
            
        # Create the random number generator instance
        self.rng = np.random.RandomState(self.seed)
        
        # Constants
        self.dt = params['dt']
        self.dt_ms = self.dt*1e3
        self.n_sigma = params['n_sigma']
        self.n_assoc = int(params['n_assoc'])
        self.tau_s = params['tau_s']
        self.n_pat = int(params['n_pat'])
        self.n_in = int(params['n_in'])
        self.H_d = params['H_d']
        self.eta_0 = params['eta']
        self.a = params['a'] if params['a']>.5 else self.rng.normal(1+params['a'],.01,self.n_assoc) 
        self.n_trial = int(params['n_trial'])
        self.t_dur = params['t_dur']; self.n_time = int(self.t_dur/self.dt)
        self.train = params['train']
        self.US = params['US']
        self.CS = params['CS']
        self.sign = params['sign']
        self.fun = params['fun']
        self.every_perc = params['every_perc']
        self.dale = params['dale']
        self.I_inh = params['I_inh']
        self.n_mem = int(params['n_mem'])
        self.n_fb = int(params['n_in'])
        self.mem_net_id = params['mem_net_id']
        self.out = params['out']
        self.CS_disap = params['CS_disap']; self.n_CS_disap = int(self.CS_disap/self.dt)
        self.US_ap = params['US_ap']; self.n_US_ap = int(self.US_ap/self.dt)
        self.US_jit = params['US_jit']; self.n_US_jit = int(self.US_jit/self.dt)
        self.est_every = params['est_every']
        self.DA_plot = params['DA_plot']
        self.trial_dyn = params['trial_dyn']
        self.flip = params['flip']
        self.extinct = params['extinct']
        self.n_wait = int(params['t_wait']/self.dt)
        self.reacquire = params['reacquire']
        self.exact = params['exact']
        self.filter = params['filter']
        self.rule = params['rule']
        self.norm = params['norm']
        self.m = params['m']
        self.no_recurrent = params['no_recurrent']
        
        # Constant inhibition, to motivate lower firing rates
        self.g_inh = 3*np.sqrt(1/self.n_assoc)
        
        # Averaging window for BCM rule
        self.T = params['T']
        self.alpha = self.dt/self.T
        
        # Load memory network, implemented by pretrained RNN
        if self.mem_net_id is not None:
            net = RNN(self.n_in,self.n_mem,self.n_in,self.n_sigma,self.tau_s,self.dt_ms,seed=None)
            checkpoint = torch.load(os.path.join(data_path,self.mem_net_id + '.pth'))
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            self.mem_net = net
            # Determine number of feedback elements
            if not self.out:
                self.n_fb = self.n_mem
        
        # Generate US and CS patterns if not available
        if self.US is None:
            self.gen_US_CS()
        
        # Weights
        if params['W_rec'] is None:
            self.W_rec = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_assoc))
            self.W_ff = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
            self.W_fb = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_fb))
            if self.dale:
                # All recurrent connections should be excitatory
                sign = np.ones(self.n_assoc); self.sign = np.diag(sign)                
                self.W_rec = np.abs(self.W_rec)
            
        else:
            self.W_rec = params['W_rec']
            self.W_ff = params['W_ff']
            self.W_fb = params['W_fb']
            self.sign = params['sign']

        # Set recurrent weights to zero if no_recurrent is True
        if params['no_recurrent']:
            self.W_rec = np.zeros((self.n_assoc, self.n_assoc))
        
        # Compute decoder matrix
        self.est_decoder()
        
    
    def simulate(self):
        # Simulation method
        start = time()
        
        # Get random trials
        trials = self.rng.choice(range(self.n_pat),self.n_trial,replace=True)
        
        # Save average errors across simulation
        batch_size = int(self.every_perc/100*self.n_trial)
        t_sampl = int(100/self.every_perc)
        self.avg_err = np.zeros((t_sampl,self.n_assoc))
        batch_num = 0
        
        # Store network estimates in single trial level
        if self.est_every:
            store_size = self.n_trial
        else:
            store_size = t_sampl
            
        self.US_est = np.zeros(tuple([store_size])+self.CS.shape)
        self.Phi_est = np.zeros(tuple([store_size])+self.Phi.shape)
        self.Phi_est_US = np.zeros(tuple([store_size])+self.Phi.shape)
        self.E = np.zeros(tuple([store_size])+(self.n_pat,))
        
        # Store expectation for entire trial to create DA release plots
        if self.DA_plot:
            self.DA_u = np.zeros((self.n_trial,self.n_time))
            
        # Store other within trial dynamics    
        if self.trial_dyn:
            self.error = np.zeros((self.n_trial,self.n_assoc,self.n_time))
            self.PSP = np.zeros((self.n_trial,self.n_assoc+self.n_fb,self.n_time))
            self.dW_rec = np.zeros((self.n_trial,self.n_assoc,self.n_assoc,self.n_time))
            self.dW_fb = np.zeros((self.n_trial,self.n_assoc,self.n_fb,self.n_time))
            self.E_tr = np.zeros((self.n_trial,self.n_time))
            self.eta = np.zeros((self.n_trial,self.n_time))
        
        # Transduction delays for perception of US
        n_trans = int(2*self.tau_s/self.dt_ms)
        
        for j, trial in enumerate(trials):
            
            # Flip CS-US associations mid-learning
            if self.flip and j == int(4*self.n_trial/5):
                if self.n_pat == 1:
                    # If only one pattern available, generate new US with required H_d
                    flip = np.zeros(self.n_in,dtype=bool)
                    flip[sample(range(self.n_in),self.H_d)] = True
                    self.US = np.concatenate((np.invert(self.US.astype(bool),where=flip).astype(float),self.US))
                    self.est_decoder()
                else:
                    self.US = self.US[::-1]
                    self.Phi = self.Phi[::-1]
                    
            # Jitter US appearance time
            if self.US_jit != 0:
                n_jit = randint(-self.n_US_jit,self.n_US_jit)
            else:
                n_jit = 0
            
            # Inputs to the network
            I_ff = np.zeros((self.n_time,self.n_in)); g_inh = np.zeros(self.n_time)
            # Determine whether in extinction phase
            if self.extinct and j > int(self.n_trial/5):
                show_US = False
            elif self.reacquire and int(self.n_trial/5) < j < int(4*self.n_trial/5):
                show_US = False
            else:
                show_US = True
                
            if show_US:
                I_ff[self.n_US_ap+n_jit:,:] = self.US[trial,:]
                g_inh[self.n_US_ap+n_jit:] = self.g_inh
                n_trigger = self.n_US_ap + n_trans + n_jit
            else:
                n_trigger = self.n_US_ap + n_trans + self.n_wait
            
            E = np.zeros(self.n_pat)
            # Store history of E to retrieve values delayed by the same delay as the US detection
            E_hist = deque(maxlen=n_trans+1)
            E_hist.append(E); E_hist.append(E)
            
            if self.mem_net_id is None:
                I_fb = np.zeros((self.n_time,self.n_fb)); I_fb[0:self.n_CS_disap,:] = self.CS[trial,:]
            else:
                inp = torch.zeros(1,self.n_time,self.n_in)
                inp[0,0:self.n_CS_disap,:] = torch.from_numpy(self.CS[trial,:]).type(torch.float)
                out, fr = self.mem_net(inp)
                if self.out:
                    I_fb = out[0,:].detach().numpy()
                else:
                    I_fb = fr[0,:].detach().numpy()
            
            # Initialize network
            r, r_m, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, C_p_u ,C_p_r, \
                                C_n_u, C_n_r = self.init_net()
            
            # Store errors after US appears, omitting transduction delays
            err = np.zeros((self.n_time-self.n_US_ap-n_trans,self.n_assoc))
            
            for i in range(1,self.n_time):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i  = assoc_net.dynamics(r,
                                I_ff[i,:],I_fb[i,:],self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt_ms,
                                self.n_sigma,g_inh[i],self.I_inh,self.fun,
                                self.a,self.tau_s)
                
                # Update time-averaged firing rate (exponential average)
                r_m = (1-self.alpha) * r_m + self.alpha * self.a * r
                
                # Estimate US
                US_est = np.dot(self.D,r)
                                
                # Form expectation
                E = self.expectation(US_est[None,:])[0][0]
                
                # Append to history and obtain delayed value
                E_del = util.update_history(E,E_hist)
                    
                # Surprise signal activates at t_trigger
                if i==n_trigger:                
                    S = self.surprise(trial,show_US,E_del)
                else:
                    S = 0
                
                # Neuromodulator concentration dynamics
                C_p_u, C_p_r, C_n_u, C_n_r = assoc_net.neuromodulator_dynamics(C_p_u,C_p_r,
                                                    C_n_u,C_n_r,S,self.dt_ms)
                
                # Learning rate
                eta = assoc_net.learn_rate(C_p_u,C_n_u,self.eta_0)
                
                # Weight modification
                if self.train:
                    self.W_rec, self.W_fb, dW_rec, dW_fb = assoc_net.learn_rule(self.W_rec,
                                self.W_fb,r,error,Delta,PSP,eta,self.dt_ms,
                                self.dale,self.sign,self.filter,self.rule,self.norm,r_m,
                                no_recurrent=self.no_recurrent)
                    if i>self.n_US_ap+n_trans:
                        err[i-self.n_US_ap-n_trans,:] = error
                
                # Save dopamine uptake at any point in trial
                if self.DA_plot:
                    self.DA_u[j,i] = C_p_u
                
                # Same trial dynamics
                if self.trial_dyn:
                    self.error[j,:,i] = error
                    self.PSP[j,:,i] = PSP
                    self.dW_rec[j,:,:,i] = dW_rec
                    self.dW_fb[j,:,:,i] = dW_fb
                    if E_del is not None:
                        self.E_tr[j,i] = E_del[trial]
                    self.eta[j,i] = eta
                    #self.US_est_tr[j,:,i] = US_est
                
            # Save network estimates after each trial
            if self.est_every:
                self.US_est[j,:], self.Phi_est[j,:] = self.est_US()
                _, self.Phi_est_US[j,:] = self.est_US(show_US=True,t_mult=2)
                self.E[j,:] = np.diag(self.expectation(self.US_est[j,:])[0])
            
            # Obtain average error at the end of every batch of trials
            if (j % batch_size == 0):
                print('{} % of the simulation complete'.format(round(j/self.n_trial*100)))
                err = np.abs(err)
                self.avg_err[batch_num,:] = np.average(err,0)
                print('Average error is {} Hz'.format(round(1000*np.average(err),2)))
                
                # Save estimates at the end of every batch
                if not self.est_every:
                    self.US_est[batch_num,:], self.Phi_est[batch_num,:] = self.est_US()
                    _, self.Phi_est_US[batch_num,:] = self.est_US(show_US=True,t_mult=2)
                    self.E[batch_num,:] = np.diag(self.expectation(self.US_est[batch_num,:])[0])
                    
                batch_num += 1
        
        # Simulation time
        end = time()
        self.sim_time = end-start
        print('Execution time: {} minutes {} seconds'.format(int(self.sim_time // 60),
                                                             int(self.sim_time % 60)))
        
    
    def gen_US_CS(self):
        # Obtain set of USs and corresponding CSs
        
        self.US, self.CS = util.gen_US_CS(self.n_pat,self.n_in,self.H_d,self.exact)
    
    
    def est_decoder(self,mode='analytic'):
        # Compute decoder of US from associative network
        
        if mode == 'pseudoinv':
            self.D = np.linalg.pinv(self.W_ff)
            
        elif mode == 'analytic':
            self.get_Phi()
            self.D = np.dot(np.linalg.pinv(self.Phi),self.US).T
    
    
    def init_net(self):
        # Initializes network to random state
        
        # initialize voltages, currents and weight updates
        V_d = self.rng.uniform(0,1,self.n_assoc); V = self.rng.uniform(0,1,self.n_assoc)
        I_d = np.zeros(self.n_assoc); Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_fb))
        PSP = np.zeros(self.n_assoc+self.n_fb); I_PSP = np.zeros(self.n_assoc+self.n_fb)
        g_e = np.zeros(self.n_assoc); g_i = np.zeros(self.n_assoc)
        r = self.rng.uniform(0,.15,self.n_assoc); C_p_u = 0; C_p_r = 0; C_n_u = 0; C_n_r = 0;
        r_m = self.rng.uniform(0,.15,self.n_assoc)
        
        return r, r_m, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, C_p_u, C_p_r, C_n_u, C_n_r
    
    
    def est_US(self,show_US=False,t_mult=5):
        # Computes estimated USs from CSs after learning
        
        # Time to settle is defined as multiple of synaptic time constant
        n_settle = int(t_mult*self.tau_s/self.dt_ms)
        
        # Initialize
        US_est = np.zeros(self.CS.shape)
        Phi_est = np.zeros((self.CS.shape[0],self.Phi.shape[1]))
        
        
        for i, CS in enumerate(self.CS):
            
            if show_US:
                I_ff = self.US[i]
            else:
                # Show just CS
                I_ff = np.zeros(self.n_in)
            
            if self.mem_net_id is None:
                I_fb = np.repeat(CS[None,:],n_settle,axis=0)
            else:
                inp = torch.from_numpy(CS).type(torch.float)
                inp = inp.repeat(1,n_settle,1)
                out, fr = self.mem_net(inp)
                if self.out:
                    I_fb = out[0,:].detach().numpy()
                else:
                    I_fb = fr[0,:].detach().numpy()
            
            # initialize network
            r, r_m, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, _, _, _, _ = self.init_net()
            
            for j in range(1,n_settle):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i = assoc_net.dynamics(r,
                                I_ff,I_fb[j,:],self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt_ms,
                                self.n_sigma,0,self.I_inh,self.fun,self.a,self.tau_s)
            
            # Decode US from firing rates of associative net
            Phi_est[i,:] = r
            US_est[i,:] = np.dot(self.D,r)
            
        return US_est, Phi_est
    
    
    def expectation(self,US_est):
        # Form an expectation of a US
        if len(US_est.shape) == 1:
            US_est = US_est[None,:]
        
        # Find adjecency to RBF kernels
        k = (8/self.H_d)**self.m
        d = np.sqrt(np.sum((US_est[:,None,:] - self.US[None,:])**2,2))
        E = np.exp(-k*d**self.m)
        
        return E, d
    
    def surprise(self,trial,show_US,E):
        
        if show_US:
            S = 1 - E[trial]
        else:
            S = - E[trial]
        
        return S
        
    
    def get_Phi(self):
        # Finds and stores steady-state firing rates for all USs
        
        self.Phi = np.zeros((self.US.shape[0],self.n_assoc))
        
        for i, US in enumerate(self.US):
            
            # US is only input to the network
            I_ff = US
            
            # Find the steady-state firing rate
            r_ss = assoc_net.ss_fr(I_ff,self.W_ff,self.g_inh,self.fun)
            
            # Store steady-state firing rate
            self.Phi[i,:] = r_ss
            

# Two associative networks class

class network2:
    
    def __init__(self,params,seed=None):
        
        # Random seed
        self.seed = seed
        
        # Set the random seed for NumPy and PyTorch
        if seed is not None:
            np.random.seed(self.seed)
            
        # Create the random number generator instance
        self.rng = np.random.RandomState(self.seed)
        
        # Constants
        self.dt = params['dt']
        self.dt_ms = self.dt*1e3
        self.n_sigma = params['n_sigma']
        self.n_assoc = int(params['n_assoc'])
        self.tau_s = params['tau_s']
        self.n_in = int(params['n_in'])
        self.eta_0 = params['eta']
        self.a = params['a'] if params['a']>.5 else self.rng.normal(1+params['a'],.01,self.n_assoc) 
        self.n_trial = int(params['n_trial'])
        self.t_dur = params['t_dur']; self.n_time = int(self.t_dur/self.dt)
        self.train = params['train']
        self.fun = params['fun']
        self.every_perc = params['every_perc']
        self.dale = params['dale']
        self.I_inh = params['I_inh']
        self.CS_2_ap_tr = params['CS_2_ap_tr']
        self.US_ap = params['US_ap']; self.n_US_ap = int(self.US_ap/self.dt)
        self.est_every = params['est_every']
        self.overexp = params['overexp']
        self.salience = params['salience']
        self.cont = params['cont']
        self.cond_dep = params['cond_dep']
        self.filter = params['filter']
        self.rule = params['rule']
        self.norm = params['norm']
        self.m = params['m']
        
        # Constant inhibition, to motivate lower firing rates
        self.g_inh = 3*np.sqrt(1/self.n_assoc)
        
        # Generate US and CS patterns
        self.US = self.rng.choice([0,1],self.n_in)
        self.CS_1 = self.salience * self.rng.choice([0,1],self.n_in)
        self.CS_2 = self.rng.choice([0,1],self.n_in)
        
        # Weights
        self.W_rec_1 = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_assoc))
        self.W_rec_2 = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_assoc))
        self.W_ff_1 = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
        self.W_ff_2 = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
        self.W_fb_1 = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
        self.W_fb_2 = self.rng.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
        if self.dale:
            # All recurrent connections should be excitatory
            sign = np.ones(self.n_assoc); self.sign = np.diag(sign)                
            self.W_rec_1 = np.abs(self.W_rec_1)
            self.W_rec_2 = np.abs(self.W_rec_2)
        else:
            self.sign = None
        
        # Compute decoder matrices
        self.est_decoders()
        
    
    def simulate(self):
        # Simulation method
        start = time()
        
        # Save average errors across simulation
        batch_size = int(self.every_perc/100*self.n_trial)
        t_sampl = int(100/self.every_perc)
        self.avg_err_1 = np.zeros((t_sampl,self.n_assoc))
        self.avg_err_2 = np.zeros((t_sampl,self.n_assoc))
        batch_num = 0
        
        # Store network estimates in single trial level
        if self.est_every:
            store_size = self.n_trial
        else:
            store_size = t_sampl    
            
        self.US_est_1 = np.zeros((store_size,self.n_in))
        self.US_est_2 = np.zeros((store_size,self.n_in))
        self.Phi_1_est = np.zeros((store_size,self.n_assoc))
        self.Phi_2_est = np.zeros((store_size,self.n_assoc))
        self.E_1 = np.zeros(store_size)
        self.E_2 = np.zeros(store_size)
            
        # Transduction delays for perception of US
        n_trans = int(2*self.tau_s/self.dt_ms)
        
        # Time of US perception
        n_trigger = self.n_US_ap + n_trans
        
        # Determine in which trials each CS is present
        if self.overexp:
            CS_1_pr = np.concatenate((np.arange(self.CS_2_ap_tr),np.arange(2*self.CS_2_ap_tr,self.n_trial)))
        else:
            CS_1_pr = np.arange(self.n_trial)
        CS_2_pr = np.arange(self.CS_2_ap_tr,self.n_trial)
        
        # Remove trials where CSs are present to test for contingency effects
        keep = self.rng.permutation(np.arange(CS_1_pr.size))[:int(CS_1_pr.size*self.cont[0])]
        keep.sort(); CS_1_pr = CS_1_pr[keep]
        if self.cond_dep:
            keep = self.rng.permutation(np.arange(CS_1_pr.size))[:int(CS_2_pr.size*self.cont[1])]
            keep.sort(); CS_2_pr = CS_1_pr[keep]
        else:
            keep = self.rng.permutation(np.arange(CS_2_pr.size))[:int(CS_2_pr.size*self.cont[1])]
            keep.sort(); CS_2_pr = CS_2_pr[keep]
        
        # Inputs to network
        I_ff = np.zeros((self.n_time,self.n_in)); I_ff[self.n_US_ap:,:] = self.US
        g_inh = np.zeros(self.n_time); g_inh[self.n_US_ap:] = self.g_inh
        I_fb_1 = np.zeros((self.n_time,self.n_in))
        I_fb_2 = np.zeros((self.n_time,self.n_in))
        
        # Expectation history
        E = 0; E_hist = deque(maxlen=n_trans+1); E_hist.append(E); E_hist.append(E)
        
        for j in range(self.n_trial):
            
            # Feedback inputs to networks
            I_fb_1[:] = self.CS_1 if j in CS_1_pr else 0
            I_fb_2[:] = self.CS_2 if j in CS_2_pr else 0
            
            # Initialize networks
            r_1, V_1, I_d_1, V_d_1, Delta_1, PSP_1, I_PSP_1, g_e_1, g_i_1, \
                C_p_u, C_p_r, C_n_u, C_n_r = self.init_net()
            r_2, V_2, I_d_2, V_d_2, Delta_2, PSP_2, I_PSP_2, g_e_2, g_i_2, \
                _, _, _, _ = self.init_net()
            
            # Store errors after US appears, omitting transduction delays
            err_1 = np.zeros((self.n_time-self.n_US_ap-n_trans,self.n_assoc))
            err_2 = np.zeros((self.n_time-self.n_US_ap-n_trans,self.n_assoc))
            
            for i in range(1,self.n_time):
                
                # One-step forward dynamics
                r_1, V_1, I_d_1, V_d_1, error_1, PSP_1, I_PSP_1, g_e_1, g_i_1 = \
                                assoc_net.dynamics(r_1,I_ff[i,:],I_fb_1[i,:],
                                self.W_rec_1,self.W_ff_1,self.W_fb_1,V_1,I_d_1,
                                V_d_1,PSP_1,I_PSP_1,g_e_1,g_i_1,self.dt_ms,
                                self.n_sigma,g_inh[i],self.I_inh,self.fun,
                                self.a,self.tau_s)
                
                r_2, V_2, I_d_2, V_d_2, error_2, PSP_2, I_PSP_2, g_e_2, g_i_2 = \
                                assoc_net.dynamics(r_2,I_ff[i,:],I_fb_2[i,:],
                                self.W_rec_2,self.W_ff_2,self.W_fb_2,V_2,I_d_2,
                                V_d_2,PSP_2,I_PSP_2,g_e_2,g_i_2,self.dt_ms,
                                self.n_sigma,g_inh[i],self.I_inh,self.fun,
                                self.a,self.tau_s)
                
                # Estimate US
                US_est_1 = np.dot(self.D_1,r_1)
                US_est_2 = np.dot(self.D_2,r_2)
                
                # Form expectation
                E_1, _ = self.expectation(US_est_1[None,:])
                E_2, _ = self.expectation(US_est_2[None,:])                    
                E = E_1 + E_2
                
                # Append to history and obtain delayed value
                E_del = util.update_history(E,E_hist)
                    
                # Surprise signal activates at t_trigger
                if i==n_trigger:                
                    S = self.surprise(E_del)
                else:
                    S = 0
                
                # Neuromodulator concentration dynamics
                C_p_u, C_p_r, C_n_u, C_n_r = assoc_net.neuromodulator_dynamics(C_p_u,C_p_r,
                                                    C_n_u,C_n_r,S,self.dt_ms)
                
                # Learning rate
                eta = assoc_net.learn_rate(C_p_u,C_n_u,self.eta_0)
                
                # Weight modification
                if self.train:
                    self.W_rec_1, self.W_fb_1, dW_rec_1, dW_fb_1 = assoc_net.learn_rule(self.W_rec_1,
                                self.W_fb_1,r_1,error_1,Delta_1,PSP_1,eta,self.dt_ms,
                                self.dale,self.sign,self.filter,self.rule,self.norm)
                    self.W_rec_2, self.W_fb_2, dW_rec_2, dW_fb_2 = assoc_net.learn_rule(self.W_rec_2,
                                self.W_fb_2,r_2,error_2,Delta_2,PSP_2,eta,self.dt_ms,
                                self.dale,self.sign,self.filter,self.rule,self.norm)
                    if i>self.n_US_ap+n_trans:
                        err_1[i-self.n_US_ap-n_trans,:] = error_1
                        err_2[i-self.n_US_ap-n_trans,:] = error_2
                        
            # Save network estimates after each trial
            if self.est_every:
                [self.US_est_1[j,:], self.US_est_2[j,:]], [self.Phi_1_est[j,:], 
                                         self.Phi_2_est[j,:]] = self.est_US()
                self.E_1[j], _ = self.expectation(self.US_est_1[j,:][None,:])
                self.E_2[j], _ = self.expectation(self.US_est_2[j,:][None,:])
            
            
            # Obtain average error at the end of every batch of trials
            if (j % batch_size == 0):
                print('{} % of the simulation complete'.format(round(j/self.n_trial*100)))
                err_1 = np.abs(err_1)
                err_2 = np.abs(err_2)
                self.avg_err_1[batch_num,:] = np.average(err_1,0)
                self.avg_err_2[batch_num,:] = np.average(err_2,0)
                print('Average error for network 1 is {} Hz'.format(round(1000*np.average(err_1),2)))
                print('Average error for network 2 is {} Hz'.format(round(1000*np.average(err_2),2)))
                
                # Save estimates at the end of every batch
                if not self.est_every:
                    [self.US_est_1[batch_num,:], self.US_est_2[batch_num,:]], \
                                [self.Phi_1_est[batch_num,:],
                                 self.Phi_2_est[batch_num,:]] = self.est_US()
                    self.E_1[batch_num,:], _ = self.expectation(self.US_est_1[batch_num,:][None,:])
                    self.E_2[batch_num,:], _ = self.expectation(self.US_est_2[batch_num,:][None,:])
                
                batch_num += 1
        
        # Simulation time
        end = time()
        self.sim_time = end-start
        print('Execution time: {} minutes {} seconds'.format(int(self.sim_time // 60),
                                                             int(self.sim_time % 60)))

            
    
    def est_decoders(self):
        # Compute decoders of US from associative networks
        
        self.get_Phis()
        self.D_1 = np.dot(np.linalg.pinv(self.Phi_1[None,:]),self.US[None,:]).T
        self.D_2 = np.dot(np.linalg.pinv(self.Phi_2[None,:]),self.US[None,:]).T
    
    
    def init_net(self):
        # Initializes network to random state
        
        # initialize voltages, currents and weight updates
        V_d = self.rng.uniform(0,1,self.n_assoc); V = self.rng.uniform(0,1,self.n_assoc)
        I_d = np.zeros(self.n_assoc); Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_in))
        PSP = np.zeros(self.n_assoc+self.n_in); I_PSP = np.zeros(self.n_assoc+self.n_in)
        g_e = np.zeros(self.n_assoc); g_i = np.zeros(self.n_assoc)
        r = self.rng.uniform(0,.15,self.n_assoc); C_p_u = 0; C_p_r = 0; C_n_u = 0; C_n_r = 0
        
        return r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, C_p_u, C_p_r, C_n_u, C_n_r
    
    
    def est_US(self,t_mult=5):
        # Computes estimated US from all CSs after learning
        
        # Time to settle is defined as multiple of synaptic time constant
        n_settle = int(t_mult*self.tau_s/self.dt_ms)
        
        # CSs are only input to the networks
        I_fb_1 = np.repeat(self.CS_1[None,:],n_settle,axis=0)
        I_fb_2 = np.repeat(self.CS_2[None,:],n_settle,axis=0)
        I_ff = np.zeros(self.n_in)
        
        # initialize networks
        r_1, V_1, I_d_1, V_d_1, Delta_1, PSP_1, I_PSP_1, g_e_1, g_i_1, _, _, _, _ = self.init_net()
        r_2, V_2, I_d_2, V_d_2, Delta_2, PSP_2, I_PSP_2, g_e_2, g_i_2, _, _, _, _ = self.init_net()
        
        for i in range(1,n_settle):
            
            # One-step forward dynamics
            r_1, V_1, I_d_1, V_d_1, error_1, PSP_1, I_PSP_1, g_e_1, g_i_1 = \
                            assoc_net.dynamics(r_1,I_ff,I_fb_1[i,:],self.W_rec_1,
                            self.W_ff_1,self.W_fb_1,V_1,I_d_1,V_d_1,PSP_1,I_PSP_1,
                            g_e_1,g_i_1,self.dt_ms,self.n_sigma,0,self.I_inh,
                            self.fun,self.a,self.tau_s)
                            
            r_2, V_2, I_d_2, V_d_2, error_2, PSP_2, I_PSP_2, g_e_2, g_i_2 = \
                            assoc_net.dynamics(r_2,I_ff,I_fb_2[i,:],self.W_rec_2,
                            self.W_ff_2,self.W_fb_2,V_2,I_d_2,V_d_2,PSP_2,I_PSP_2,
                            g_e_2,g_i_2,self.dt_ms,self.n_sigma,0,self.I_inh,
                            self.fun,self.a,self.tau_s)
        
        # Decode US from firing rates of associative nets
        Phi_1_est = r_1
        Phi_2_est = r_2
        US_est_1 = np.dot(self.D_1,r_1)
        US_est_2 = np.dot(self.D_2,r_2)
        
        return [US_est_1, US_est_2], [Phi_1_est, Phi_2_est]
    
    
    def expectation(self,US_est):
        # Form an expectation of a US
        
        # Find adjecency to RBF kernel
        k = 1
        d = np.sqrt(np.sum((US_est - self.US)**2,1))
        E = np.exp(-k*d**self.m)
        
        return E, d
    
    
    def surprise(self,E):
        
        return 1 - E
        
    
    def get_Phis(self):
        # Finds and stores steady-state firing rates for US
        
        # US is only input to the network
        I_ff = self.US
        
        # Find the steady-state firing rate
        r_ss_1 = assoc_net.ss_fr(I_ff,self.W_ff_1,self.g_inh,self.fun)
        r_ss_2 = assoc_net.ss_fr(I_ff,self.W_ff_2,self.g_inh,self.fun)
        
        # Store steady-state firing rate
        self.Phi_1 = r_ss_1
        self.Phi_2 = r_ss_2


# RNN class

class RNN(nn.Module):
    
    def __init__(self,inp_size,rec_size,out_size,n_sd=.1,tau=100,dt=10,seed=None):
        super().__init__()
        
        # Manual seed
        self.seed = seed
        
        # Set seed for PyTorch
        if seed is not None:
            torch.manual_seed(self.seed)        
            # Generator
            self.rng = torch.Generator()
        
        
        # Constants
        self.inp_size = inp_size
        self.rec_size = rec_size
        self.n_sd = n_sd
        self.tau = tau
        self.alpha = dt / self.tau
        
        # Layers
        self.inp_to_rec = nn.Linear(inp_size, rec_size)
        self.rec_to_rec = nn.Linear(rec_size, rec_size)
        self.rec_to_out = nn.Linear(rec_size, out_size)
        

    def init(self,inp_shape):
        # Initializes network activity to zero
        
        n_batch = inp_shape[0]
        r = torch.zeros(n_batch,self.rec_size)
        
        return r


    def dynamics(self,inp,r):
        # Defines dynamics of the network
        
        if self.seed is None:
            e = torch.randn(self.rec_size)
        else:
            e = torch.randn(self.rec_size, generator=self.rng)
            
        h = self.inp_to_rec(inp) + self.rec_to_rec(r) + self.n_sd*e
        
        r_new = (1 - self.alpha)*r + self.alpha*torch.relu(h)
        
        return r_new


    def forward(self,inp):
        # Forward pass
        
        # Initialize network
        r = self.init(inp.shape)
        out = []; fr = []
        
        # Simulate
        for i in range(inp.shape[1]):
            r = self.dynamics(inp[:,i],r)
            # Store network output and activity for entire batch
            fr.append(r)
            out.append(self.rec_to_out(r))
            
        fr = torch.stack(fr, dim=1)
        out = torch.stack(out, dim=1)
        
        return out, fr
    
    
    def reset_params(self):
        # Reset everything in the network
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()