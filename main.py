"""
Main simulation classes.
"""

import numpy as np
import util
import assoc_net


# Main simulation class

class simulation:
    
    def __init__(self,params):
        
        # Constants
        self.dt = params['dt']
        self.dt_ms = self.dt*1e3
        self.n_sigma = params['dt']
        self.n_assoc = int(params['n_assoc'])
        self.tau_s = params['tau_s']
        self.n_pat = int(params['n_pat'])
        self.n_in = int(params['n_in'])
        self.H_d = params['H_d']
        self.eta = params['eta']
        self.n_trial = int(params['n_trial'])
        self.t_dur = params['t_dur']
        self.n_time = int(self.t_dur/self.dt)
        self.train = params['train']
        self.US = params['US']
        self.CS = params['CS']
        
        # Weights
        if params['w_rec'] is None:
            self.w_rec = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_assoc))
            self.w_ff = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
            self.w_fb = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
        else:
            self.w_rec = params['w_rec']
            self.w_ff = params['w_ff']
            self.w_fb = params['w_fb']
       
        
    def simulate(self):
        
        # Obtain set of US and corresponding CS
        if self.US is None:
            self.US, self.CS = util.gen_US_CS(self.n_pat,self.n_in,self.H_d)
        
        # Get random trials
        trials = np.random.choice(range(self.n_pat),self.n_trial,replace=True)
        
        # Simulate trial by trial
        in_ff = np.empty((self.n_time,self.n_in))
        in_fb = np.empty((self.n_time,self.n_in))
        
        for trial in trials:
            
            # Inputs to the network
            in_ff[:] = self.US[trial,:]
            in_fb[:] = self.CS[trial,:]
            
            # initialize voltages, currents and weight updates
            V_d = np.random.uniform(0,1,self.n_assoc); V = np.random.uniform(0,1,self.n_assoc)
            I_d = np.zeros(self.n_assoc); Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_in))
            PSP = np.zeros(self.n_assoc+self.n_in); I_PSP = np.zeros(self.n_assoc+self.n_in)
            r = np.random.uniform(0,.15,self.n_assoc)
            
            for i in range(self.n_time-1):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP = assoc_net.dynamics(r,
                                in_ff[i,:],in_fb[i,:],self.w_rec,self.w_ff,
                                self.w_fb,V,I_d,V_d,PSP,I_PSP,self.dt,self.n_sigma)
                
                # Weight modification
                if self.train:
                    self.w_rec, self.w_fb = assoc_net.learn_rule(self.w_rec,
                                    self.w_fb,error,Delta,PSP,self.eta,self.dt)