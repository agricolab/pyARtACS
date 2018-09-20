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
def _modify_kernel_mode(kernel:np.array, mode:str='uniform'):
    #kernel = kernel.copy()
    width = (kernel[:kernel.shape[0]//2]<0).sum()
    period = int(np.unique(np.diff(np.where(kernel!=0)[0])))
    if mode=='uniform':
        pass
    elif 'exp' in mode:
        weights = signal.exponential(((width-1)*2)+1)[:width]
        weights /= weights.sum() * 2      
        weights = np.hstack((-weights, 1.0, -weights[::-1]))
        kernel[::period] = weights        

    else:
        raise NotImplementedError
    
    # the kernel should add up to zero
    assert np.isclose(kernel.sum(), 0.0)
    return kernel    
    
def _modify_kernel_direction(kernel:np.array, direction:str='symmetric'):
    #kernel = kernel.copy()
    if direction == 'causal' or direction == 'left':
        kernel[kernel.shape[0]//2+1:] = 0
        kernel[:kernel.shape[0]//2-1] *= 2        
    elif direction == 'right':
        kernel[kernel.shape[0]//2+1:] *= 2
        kernel[:kernel.shape[0]//2-1] = 0        
    elif 'sym' in direction:
        midpoint = kernel.shape[0]//2
        right_half = kernel[midpoint+1:].copy()
        left_half = kernel[0:midpoint].copy()
        
        if not  np.all(np.isclose(left_half, right_half[::-1])):
            kernel[midpoint+1:] += left_half[::-1]
            kernel[0:midpoint] += right_half[::-1]
            kernel[midpoint+1:] /= 2
            kernel[0:midpoint] /= 2
    else:
        raise NotImplementedError('Direction unknown')
        
    # the kernel should add up to zero
    assert np.isclose(kernel.sum(), 0.0)
    return kernel  
        
    
def _create_uniform_symmetric_kernel(freq:int, fs:int, width:int):
    '''Create a uniform symmetric comb kernel
    '''
    in_period = fs/freq    
    period = int(np.ceil(in_period))   
    if in_period != period:
        raise ValueError('Only integer periods are allowed.' + 
                         'Try resampling your signal to higher sampling rate')
    
    k = np.zeros((period*width*2)+1)
    k[0::period] = -1/(width*2)
    k[width*period] = 1
    return k

def create_kernel(freq:int, fs:int, width:int, 
                  mode:str='uniform', direction:str='symmetric'):
    kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    kernel = _modify_kernel_mode(kernel, mode)
    kernel = _modify_kernel_direction(kernel, direction)
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
        
    
    