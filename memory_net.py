"""
Train RNN to maintain memory of CS.
"""


# imports
import torch
import torch.nn as nn

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
        
        batch_size = inp_shape[0]; n_t = inp_shape[1]
        r = torch.zeros(batch_size,n_t,self.rec_size)
        out = torch.zeros(batch_size,n_t,self.rec_size)
        
        return r, out


    def rec_dynamics(self,inp,r):
        # Defines recurrent dynamics in the network
        
        h = self.inp_to_rec(inp) + self.rec_to_rec(r) + \
                    self.n_sd*torch.randn(self.rec_size)
        r += - self.alpha*r + self.alpha*torch.relu(h)
        
        return r


    def forward(self,inp):
        # Forward pass through the network
        
        r, out = self.init_rec(inp.shape)

        for i in range(1,inp.shape[1]):
            r[:,i] = self.rec_dynamics(inp[:,i],r[:,i-1])
            # Store network output and recurrent activity for entire batch
            out[:,i] = self.rec_to_out(r[:,i])

        return out, r
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()