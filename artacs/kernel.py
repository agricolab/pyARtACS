#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:29:48 2018

@author: rgugg
"""
import numpy as np
from scipy import signal
from artacs.tools import resample_by_fs, resample_by_count
# %%
def _estimate_prms_from_kernel(kernel):
    '''estimate period and width from kernel
    
    args
    ----
    kernel:array
        kernel from which we estimate the parameters
        
    returns
    -------
    period:int
        distance between two weight values in samples
    width:int
        number of weight values in each direction
    direction:str {'left', 'right', 'sym'}
        direction of kernel

    '''
  
    
    period = np.unique(np.diff(np.where(kernel!=0)[0]))
    if len(period) != 1:
        raise ValueError('Multiple or no periods recognized in kernel' + 
                         'Was this kernel correctly constructed?')
    
    period = int(period)
    left_width = (kernel[:kernel.shape[0]//2]<0).sum()
    right_width = (kernel[kernel.shape[0]//2:]<0).sum()        
    
    if left_width == 0:
        direction = 'right'
        width = right_width
    elif right_width == 0:
        direction = 'left'
        width = left_width
    elif right_width == left_width:
        direction = 'sym'
        width = left_width # both need to be equal, so it doesn't matter
    else:
        raise ValueError('Kernel presents with unclear direction.' + 
                         'Was this kernel correctly constructed?')
 
    return period, width, direction

# %%
def _weigh_exp(width):
    'create exponential weights'    
    weights = signal.exponential(((width-1)*2)+1)[:width]
    weights /= (weights.sum())    
    return weights

def _weigh_linear(width):
    'create linear weights'    
    weights = np.linspace(0, 1, num=width)
    weights /= (weights.sum())    
    return weights

def _weigh_gaussian(width, sigma = 1):
    'create gaussian weights'    
    weights = signal.gaussian(width*2, sigma)[0:width]
    weights /= ( weights.sum())
    return weights

def _weigh_uniform(width):
    'create uniform weights'    
    weights = np.ones(width)
    weights /= ( weights.sum())
    return weights

def _weigh_not(width):
    'create zero weights'
    weights = np.zeros(width)
    return weights

def create_kernel(freq:int, fs:int, width:int, 
                  left_mode:str='uniform', right_mode:str='uniform'):
    
    in_period = fs/freq    
    period = int(np.ceil(in_period))   
    if in_period != period:
        raise ValueError('Only integer periods are allowed.' + 
                         'Try resampling your signal to higher sampling rate')
    
    weighfoos = {'uniform':_weigh_uniform,
                 'uni':_weigh_uniform,
                 'none':_weigh_not,
                 'zero':_weigh_not,
                 'gauss':_weigh_gaussian,
                 'normal':_weigh_gaussian,
                 'linear':_weigh_linear,
                 'exp':_weigh_exp,
                 'exponential':_weigh_exp
                 }
    
    left_weights = weighfoos[left_mode.lower()](width)
    right_weights = weighfoos[right_mode.lower()](width)[::-1]
    norm = left_weights.sum() + right_weights.sum()
    
    weights = np.hstack((-left_weights/norm, 1.0, -right_weights/norm))
    midpoint = period*width    
    kernel = np.zeros((midpoint*2)+1)
    kernel[::period] = weights   
    return kernel

# %%
def kernel_filter(indata, freq:int, fs:int, width:int):        
    
    in_channels = indata.shape[0]
    in_samples = indata.shape[1]
    in_period = fs/freq
    resample_flag = in_period != int(fs/freq)
    
    if resample_flag:
        old_fs = fs
        period =int(np.ceil(in_period))
        fs = int(period * freq)
        #data = signal.resample(indata, int(in_samples/old_fs)*fs, axis=1)
        outdata = []
        for chan_idx, chan in enumerate(indata):
            outdata.append(resample_by_fs(chan, up=fs, down = old_fs))        
        data = np.asanyarray(outdata)
    else:
        data = indata
        period = int(in_period)        
    #-------------------------------------------------------------------------
    
    k = np.zeros((period*width*2)+1)
    k[0::period] = -1/(width*2)
    k[width*period] = 1
    pad_width = (period*width)
    
    fdata = np.zeros((in_channels, data.shape[1]))    
    for idx, c in enumerate(data):
        # pad 
        padded_channel = np.pad(c, pad_width, 'wrap')
        filt_channel = np.convolve(padded_channel, k, 'same')
        # remove padding
        fdata[idx,:] = filt_channel[pad_width:-pad_width]
        
    
    #-------------------------------------------------------------------------
    if resample_flag:
        filtered = []
        for chan_idx, chan in enumerate(fdata):
            filtered.append(resample_by_count(chan, in_samples))        
        filtered = np.asanyarray(filtered)
        
    else:
        filtered = fdata            
        
    return filtered
#%%
class KernelFilter():
    
    def __init__(self, frequency:int, fs:int, width:int, mode:str='uniform', direction:str='symmetric'):
        self.kernel = create_kernel(frequency, fs, width, mode, direction)
        # create_kernel checks whether frequency and fs are valid amd throws 
        # an exception if not. Becuase this stops __init__, an additional 
        # check is therefore not required
        self.frequency = frequency
        self.fs = fs
    
    def filter(self, signal:np.array):
        pass
        
    
    