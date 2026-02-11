import os.path as op
import random
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import datetime

def make_samples(H, mu, sigma, cutoff, n_samples):

    state = int(np.random.rand(1)[0]>0.5)
    samples = np.zeros(n_samples)
    states = np.zeros(n_samples)
    for i in range(n_samples):

        # check if state flipped:
        if np.random.rand(1)[0] < H:
            if state == 0:
                state = 1
            elif state == 1:
                state = 0
        
        # draw sample:
        sample = np.random.normal(mu[state], sigma[state], 1)[0]
        
        # threshold:
        if sample < cutoff[0]:
            sample = cutoff[0]
        elif sample > cutoff[1]:
            sample = cutoff[1]

        # add
        samples[i] = sample
        states[i] = state
    states[states==0] = -1

    LLRin = np.log(sp.stats.norm.pdf(samples,mu[0]*-1,sigma[0])/
                sp.stats.norm.pdf(samples,mu[1]*-1,sigma[1]))

    return states, samples, LLRin

def into_logspaced_freqs(values, values_min, values_max, min_f, nr_octaves):
    
    # express into octaves:
    span = values_max - values_min
    values_scaled = (values-values_min) / span * 100
    all_freqs = np.logspace(start=np.log10(min_f),
                            stop=np.log10(min_f*(2**nr_octaves)),
                            num=10000)
    freqs = np.percentile(all_freqs,values_scaled)
    return freqs


cutoff = 60
states, freqs, LLRin = make_samples(H=H, 
                            mu = [-20,20], 
                            sigma = [15,15],
                            cutoff= [-cutoff,cutoff], 
                            n_samples=1000)