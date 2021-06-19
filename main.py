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

# File directory
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'

# Main simulation class

class network:
    
    def __init__(self,params):
        
        # Constants
        self.dt = params['dt']
        self.dt_ms = self.dt*1e3
        self.n_sigma = params['n_sigma']
        self.n_assoc = int(params['n_assoc'])
        self.tau_s = params['tau_s']
        self.n_pat = int(params['n_pat'])
        self.n_in = int(params['n_in'])
        self.H_d = params['H_d']
        self.eta = params['eta']
        self.n_trial = int(params['n_trial'])
        self.t_dur = params['t_dur']; self.n_time = int(self.t_dur/self.dt)
        self.train = params['train']
        self.US = params['US']
        self.CS = params['CS']
        self.R = params['R']
        self.S = params['S']
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
        self.est_every = params['est_every']
        
        # Shunting inhibition, to motivate lower firing rates
        self.g_sh = 3*np.sqrt(1/self.n_assoc)
        
        # Memory network implemented by RNN
        if self.mem_net_id is not None:
            net = RNN(self.n_in,self.n_mem,self.n_in,self.n_sigma,self.tau_s,self.dt_ms)
            checkpoint = torch.load(data_path + self.mem_net_id + '.pth')
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            self.mem_net = net
            # Determine number of feedback elements
            if not self.out:
                self.n_fb = self.n_mem
        
        # Generate US and CS patterns if not available
        if self.US is None:
            self.gen_US_CS()
            self.R = np.random.uniform(low=.5,high=1,size=self.n_pat)
        
        # Weights
        if params['W_rec'] is None:
            self.W_rec = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_assoc))
            self.W_ff = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
            self.W_fb = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_fb))
            if self.dale:
                # 20 % inhibitory, 80 % excitatory
                S = np.ones(self.n_assoc); S[-int(self.n_assoc*.2):] = -1
                self.S = np.diag(S)
                self.W_rec = np.dot(np.abs(self.W_rec),self.S)
            
        else:
            self.W_rec = params['W_rec']
            self.W_ff = params['W_ff']
            self.W_fb = params['W_fb']
            self.S = params['S']
        
        # Compute decoder matrix
        self.est_decoder()
        
    
    def simulate(self):
        # Simulation method
        start = time()
        
        # Get random trials
        trials = np.random.choice(range(self.n_pat),self.n_trial,replace=True)
        
        # Save average errors across simulation
        batch_size = int(self.every_perc/100*self.n_trial)
        t_sampl = int(100/self.every_perc)
        self.avg_err = np.zeros((t_sampl,self.n_assoc))
        batch_num = 0
        
        # Store network estimates in single trial levels
        if self.est_every:
            self.US_est = np.zeros(tuple([self.n_trial])+self.CS.shape)
            self.Phi_est = np.zeros(tuple([self.n_trial])+self.Phi.shape)
            self.R_est = np.zeros(tuple([self.n_trial])+self.R.shape)
        
        # Transduction delays for perception of reward
        n_trans = int(3*self.tau_s/self.dt_ms)
        
        for j, trial in enumerate(trials):
            
            # Inputs to the network
            I_ff = np.zeros((self.n_time,self.n_in)); I_ff[self.n_US_ap:,:] = self.US[trial,:]
            R = np.zeros(self.n_time); R[self.n_US_ap+n_trans] = self.R[trial]
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
            r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, DA_u, DA_r = self.init_net()
            
            # Store errors after US appears, omitting transduction delays
            err = np.zeros((self.n_time-self.n_US_ap-n_trans,self.n_assoc))
            
            for i in range(1,self.n_time):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i  = assoc_net.dynamics(r,
                                I_ff[i,:],I_fb[i,:],self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt_ms,
                                self.n_sigma,self.g_sh,self.I_inh,self.fun,self.tau_s)
                
                # Estimate US
                US_est = np.dot(self.D,r)
                
                # Estimate reward
                R_est, _ = self.est_R(np.expand_dims(US_est,axis=0))
                
                # Diffuse dopamine signal dynamics
                DA_u, DA_r = assoc_net.DA_dynamics(DA_u,DA_r,R[i]-R_est,self.dt_ms)
                
                # Learning rate
                eta = assoc_net.learn_rate(DA_u,self.eta)
                
                # Weight modification
                if self.train:
                    self.W_rec, self.W_fb = assoc_net.learn_rule(self.W_rec,self.W_fb,
                                    error,Delta,PSP,eta,self.dt_ms,self.dale,self.S)
                    if i>self.n_US_ap+n_trans:
                        err[i-self.n_US_ap-n_trans,:] = error
                        
            # Save network estimates after each trial
            if self.est_every:
                self.US_est[j,:], self.Phi_est[j,:] = self.est_US()
                self.R_est[j,:], _ = self.est_R(self.US_est[j,:])
            
            # Obtain average error every batch_size trials
            if (j % batch_size == 0):
                print('{} % of the simulation complete'.format(round(j/self.n_trial*100)))
                err = np.abs(err)
                self.avg_err[batch_num,:] = np.average(err,0)
                print('Average error is {} Hz'.format(round(1000*np.average(err),2)))
                batch_num += 1
        
        end = time()
        self.sim_time = round((end-start)/3600,2)
        print("The simulation ran for {} hours".format(self.sim_time))
        
        # Final network estimates
        if not self.est_every:
            self.US_est, self.Phi_est = self.est_US()
            self.R_est, _ = self.est_R(self.US_est)
        
    
    def gen_US_CS(self):
        # Obtain set of US and corresponding CS
        
        self.US, self.CS = util.gen_US_CS(self.n_pat,self.n_in,self.H_d)
    
    
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
        V_d = np.random.uniform(0,1,self.n_assoc); V = np.random.uniform(0,1,self.n_assoc)
        I_d = np.zeros(self.n_assoc); Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_fb))
        PSP = np.zeros(self.n_assoc+self.n_fb); I_PSP = np.zeros(self.n_assoc+self.n_fb)
        g_e = np.zeros(self.n_assoc); g_i = np.zeros(self.n_assoc)
        r = np.random.uniform(0,.15,self.n_assoc); DA_u = 0; DA_r = 0 
        
        return r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, DA_u, DA_r
    
    
    def est_US(self,t_mult=5):
        # Computes estimated USs from all CSs after learning
        
        # Time to settle is defined as multiple of synaptic time constant
        n_settle = int(t_mult*self.tau_s/self.dt_ms)
        
        US_est = np.zeros(self.CS.shape)
        Phi_est = np.zeros(self.Phi.shape)
        I_ff = np.zeros(self.n_in)
        
        for i, CS in enumerate(self.CS):
            
            # CS is only input to the network
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
            r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i, _, _ = self.init_net()
            
            for j in range(1,n_settle):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i = assoc_net.dynamics(r,
                                I_ff,I_fb[j,:],self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt_ms,
                                self.n_sigma,0,self.I_inh,self.fun,self.tau_s)
            
            # Decode US from firing rates of associative net
            Phi_est[i,:] = r
            US_est[i,:] = np.dot(self.D,r)
            
            return US_est, Phi_est
            
    
    def est_R(self,US_est):
        # Estimate predicted reward
        
        k = (8/self.H_d)**2
        d = np.sqrt(np.sum((US_est - self.US)**2,1))
        R_est = np.dot(self.R,np.exp(-k*d**2))
        
        return R_est, d
        
            
    def get_Phi(self):
        # Finds and stores steady-state firing rates for all USs
        
        self.Phi = np.zeros((self.n_pat,self.n_assoc))
        
        for i, US in enumerate(self.US):
            
            # US is only input to the network
            I_ff = US
            
            # Find the steady-state firing rate
            r_m = assoc_net.ss_fr(I_ff,self.W_ff,self.g_sh,self.fun)
            
            # Store steady-state firing rate
            self.Phi[i,:] = r_m
            

# RNN class

class RNN(nn.Module):
    
    def __init__(self,inp_size,rec_size,out_size,n_sd=.1,tau=100,dt=10):
        super().__init__()
        
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
        # Initializes network
        
        n_batch = inp_shape[0]
        r = torch.zeros(n_batch,self.rec_size)
        
        return r


    def rec_dynamics(self,inp,r):
        # Defines recurrent dynamics in the network
        
        h = self.inp_to_rec(inp) + self.rec_to_rec(r) + \
                    self.n_sd*torch.randn(self.rec_size)
        r_new = (1 - self.alpha)*r + self.alpha*torch.relu(h)
        
        return r_new


    def forward(self,inp):
        # Forward pass through the network
        
        r = self.init(inp.shape)
        
        out = []; fr = []
        for i in range(inp.shape[1]):
            r = self.rec_dynamics(inp[:,i],r)
            # Store network output and recurrent activity for entire batch
            fr.append(r)
            out.append(self.rec_to_out(r))
            
        fr = torch.stack(fr, dim=1)
        out = torch.stack(out, dim=1)
        
        return out, fr
    
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()