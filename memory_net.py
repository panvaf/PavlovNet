"""
Train RNN to maintain memory of CS.
"""

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path

# Constants
n_neu = 128         # number of recurrent neurons
n_batch = 100       # size of training batch
n_iter = 1e4        # number of batches
n_in = 20           # size of patterns
t_dur = 3           # duration of trial
CS_mindur = .5      # minimum duration of CS presentation
CS_maxdur = 2       # maximum duration of CS presentation
dt = 1e-2           # step size
tau = 1e-1          # neuronal time constant (synaptic+membrane)
n_sd = .1           # standard deviation of injected noise
print_every = int(n_iter/100)
n_t = int(t_dur/dt); n_CS_mindur = int(CS_mindur/dt)
n_CS_maxdur = int(CS_maxdur/dt); n_grace = 2*int(tau/dt)

# Save location
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
net_file = 'MemNet' + str(n_neu) + \
            (('insz' + str(n_in)) if n_in != 20 else '') + \
            (('tdur' + str(t_dur)) if t_dur != 2 else '') + \
            (('n_iter' + str(n_iter)) if not n_iter==1e4 else '') + \
            (('Noise' + str(n_sd)) if n_sd else '') + \
            (('tau' + str(1e3*tau)) if tau != 1e-1 else '')

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

# Weighted mean squared error loss
        
def MSELoss_weighted(output,target,mask):
    loss = torch.sum(mask*(output - target)**2)
    size = torch.numel(target)
    norm = torch.sum(mask)
    
    avg_loss = loss/size
    normed_loss = loss/norm
    
    return avg_loss, normed_loss


# Initialize RNN  
net = RNN(n_in,n_neu,n_in,n_sd,tau*1e3,dt*1e3)

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train RNN
running_loss = 0; k = 0
train_loss = np.zeros(100)

for i in range(int(n_iter)):
    # Generate data for current batch
    n_CSs = np.random.randint(n_CS_mindur,n_CS_maxdur,n_batch)
    CS = torch.randint(low=0, high=2,size=(n_batch,1,n_in))
    inputs = CS.repeat(1,n_t,1).float()
    for j, n_CS in enumerate(n_CSs):
        inputs[j,n_CS:,:] = 0
    target = CS.repeat(1,n_t,1)
    mask = torch.ones(n_batch,n_t,n_in); mask[:,0:n_grace,:] = 0

    # Training loop
    optimizer.zero_grad()   # zero the gradient buffers
    output, fr = net(inputs)
    _, normed_loss = MSELoss_weighted(output,target,mask)
    normed_loss.backward()
    optimizer.step()    # Does the update
    
    running_loss += normed_loss.item()
    if (i % print_every == 0):
        running_loss /= print_every
        print('{} % of the simulation complete'.format(round(i/n_iter*100)))
        print('Loss {:0.3f}'.format(running_loss))
        train_loss[k] = running_loss
        running_loss = 0; k += 1

# Save network
torch.save({'state_dict': net.state_dict(),'train_loss': train_loss},
                    data_path + net_file + '.pth')