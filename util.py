"""
Utilities.
"""

import numpy as np
from scipy.spatial.distance import hamming

# Logistic activation function
    
def logistic(x,x0=1,k=1,b=2.5,s=.15):
    # s: maximum firing rate in kHz
    # x0: 50 % firing rate point
    # b: steepness of gain
    
    return s/(1+k*np.exp(-b*(x-x0)))


# Create sets of USs and corresponding CSs
    
def gen_US_CS(n_pat,n_in,H_d):
    # n_pat: number of US-CS patterns we want to associate
    # n_in: size of binary number representing each pattern
    # H_d: minimal acceptable Hamming distance between any two patterns of
    #      the same type. Current algo is simple, but effective. Could maximize
    #      distance with Reed-Solomon code, however not very biological..
    
    patterns = np.empty((2,n_pat,n_in))
    
    for st in range(2):
        for i in range(n_pat):
            while True:
                sw = 0
                patt = np.random.randint(0,2,n_in)
                # Make sure Hamming distance with existing patterns is acceptable.
                # Algo is greedy, not trying to spread codewords evenly.
                for j in range(i):
                    h_d = hamming(patt,patterns[st,j,:])*n_in
                    if h_d < H_d:
                        # Patterns too close
                        sw = 1
                        break
                # If new pattern is spaced apart from others, break while loop
                if not sw:
                    break
            patterns[st,i,:] = patt
    
    US = patterns[0,:]
    CS = patterns[1,:]
    
    return US, CS