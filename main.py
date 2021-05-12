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
        self.fun = params['fun']
        # Shunting inhibition, to motivate lower firing rates
        self.g_sh = 2*np.sqrt(1/self.n_assoc)
        
        # Weights
        if params['W_rec'] is None:
            self.W_rec = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_assoc))
            self.W_ff = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
            self.W_fb = np.random.normal(0,np.sqrt(1/self.n_assoc),(self.n_assoc,self.n_in))
        else:
            self.W_rec = params['W_rec']
            self.W_ff = params['W_ff']
            self.W_fb = params['W_fb']
       
        
    def simulate(self):
        # Simulation method
        
        # Generate US and CS patterns if not available
        if self.US is None:
            self.gen_US_CS()
        
        # Get random trials
        trials = np.random.choice(range(self.n_pat),self.n_trial,replace=True)
        
        # Simulate trial by trial
        I_ff = np.empty((self.n_time,self.n_in))
        I_fb = np.empty((self.n_time,self.n_in))
        
        for trial in trials:
            
            # Inputs to the network
            I_ff[:] = self.US[trial,:]
            I_fb[:] = self.CS[trial,:]
            
            # initialize network
            r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i = self.init_net()
            
            for i in range(self.n_time-1):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i  = assoc_net.dynamics(r,
                                I_ff[i,:],I_fb[i,:],self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt,
                                self.n_sigma, self.inh, self.fun)
                
                # Weight modification
                if self.train:
                    self.W_rec, self.W_fb = assoc_net.learn_rule(self.W_rec,
                                    self.W_fb,error,Delta,PSP,self.eta,self.dt)
    
    def gen_US_CS(self):
        # Obtain set of US and corresponding CS
        
        self.US, self.CS = util.gen_US_CS(self.n_pat,self.n_in,self.H_d)
    
    def get_decoder(self,mode='analytic'):
        # Compute decoder of US from associative network
        
        if mode == 'pseudoinv':
            self.D = np.linalg.pinv(self.W_ff)
            
        elif mode == 'analytic':
            self.ss_fr()
            self.D = np.dot(np.linalg.pinv(self.Phi),self.US)
    
    
    def init_net(self):
        # Initializes network to random state
       
        # initialize voltages, currents and weight updates
        V_d = np.random.uniform(0,1,self.n_assoc); V = np.random.uniform(0,1,self.n_assoc)
        I_d = np.zeros(self.n_assoc); Delta = np.zeros((self.n_assoc,self.n_assoc+self.n_in))
        PSP = np.zeros(self.n_assoc+self.n_in); I_PSP = np.zeros(self.n_assoc+self.n_in)
        g_e = np.zeros(self.n_assoc); g_i = np.zeros(self.n_assoc)
        r = np.random.uniform(0,.15,self.n_assoc)
        
        return r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i
    
    
    def est_US(self,t_mult=5):
        # Computes estimated USs from all CSs after learning
        
        # Compute decoder matrix
        if not hasattr(self,'D'):
            self.get_decoder()
        
        # Time to settle is defined as multiple of synaptic time constant
        n_settle = int(t_mult*self.tau_s/self.dt_ms)
        
        self.US_est = np.zeros(self.CS.shape)
        I_ff = np.zeros(self.n_in)
        
        for i, CS in enumerate(self.CS):
            
            # CS is only input to the network
            I_fb = CS
            
            # initialize network
            r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i = self.init_net()
            
            for j in range(n_settle-1):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i = assoc_net.dynamics(r,
                                I_ff,I_fb,self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt,
                                self.n_sigma, self.exc, self.fun)
            
            # Decode US from firing rates of associative net
            self.US_est[i,:] = np.dot(self.D,r)
            
    def ss_fr(self,t_mult=5):
        # Finds steady-state firing rates for all USs
        
        # Time to settle is defined as multiple of synaptic time constant
        n_settle = int(t_mult*self.tau_s/self.dt_ms)
        
        self.Phi = np.zeros(self.n_pat,self.n_assoc)
        I_fb = np.zeros(self.n_in)
        
        for i, US in enumerate(self.US):
            
            # US is only input to the network
            I_ff = US
            
            # initialize network
            r, V, I_d, V_d, Delta, PSP, I_PSP, g_e, g_i = self.init_net()
            
            for j in range(n_settle-1):
                
                # One-step forward dynamics
                r, V, I_d, V_d, error, PSP, I_PSP, g_e, g_i = assoc_net.dynamics(r,
                                I_ff,I_fb,self.W_rec,self.W_ff,
                                self.W_fb,V,I_d,V_d,PSP,I_PSP,g_e,g_i,self.dt,
                                self.n_sigma, self.exc, self.fun)
            
            # Save steady-state firing rate
            self.Phi[i,:] = r