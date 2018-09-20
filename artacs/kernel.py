#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:29:48 2018

@author: rgugg
"""
import numpy as np
from typing import Tuple
from numpy import ndarray
from warnings import warn
from scipy import signal
from artacs.tools import resample_by_fs, resample_by_count
# %%
def _estimate_prms_from_kernel(kernel:ndarray) -> Tuple[int, int]:
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

    '''
  
    
    period = np.unique(np.diff(np.where(kernel!=0)[0]))
    if len(period) != 1:
        raise ValueError('Multiple or no periods recognized in kernel' + 
                         'Was the kernel correctly constructed?')
    
    period = int(period)
    left_width = (kernel[:kernel.shape[0]//2]<0).sum()
    right_width = (kernel[kernel.shape[0]//2:]<0).sum()        
    
    if left_width == 0:
        width = right_width
    elif right_width == 0:
        width = left_width
    elif right_width == left_width:
        width = left_width # both need to be equal, so it doesn't matter
    else:
        raise ValueError('Kernel presents with unclear direction.' + 
                         'Was the kernel correctly constructed?')
 
    return period, width

# %%
def _weigh_exp(width:int) -> ndarray:
    'create exponential weights'    
    weights = signal.exponential(((width-1)*2)+1)[:width]
    weights /= (weights.sum())    
    return weights

def _weigh_linear(width:int)  -> ndarray:
    'create linear weights'    
    weights = np.linspace(1/width, 1, num=width)
    weights /= (weights.sum())    
    return weights

def _weigh_gaussian(width:int, sigma:float=1) -> ndarray:
    'create gaussian weights'    
    weights = signal.gaussian(width*2, sigma)[0:width]
    weights /= ( weights.sum())
    return weights

def _weigh_uniform(width:int) -> ndarray:
    'create uniform weights'    
    weights = np.ones(width)
    weights /= ( weights.sum())
    return weights

def _weigh_not(width:int) -> ndarray:
    'create zero weights'
    weights = np.zeros(width)
    return weights

def create_kernel(freq:int, fs:int, width:int, 
                  left_mode:str='uniform', 
                  right_mode:str='uniform')  -> ndarray:
    
    in_period = fs/freq    
    period = int(np.ceil(in_period))   
    if in_period != period:
        warn ('Only integer periods are natively supported.' + 
              'Try resampling to higher sampling rate')
        fs = int(period * freq)             
    
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
def _filter_1d(indata, fs:int, freq:int, kernel:ndarray):
    
    in_samples = indata.shape[0]
    in_period = fs/freq
        
    #if sampling rate of signal and artifact are not integer divisible,
    # we have to resample the data
    resample_flag = ( in_period != int(np.ceil(in_period)) )     
    if resample_flag:
        old_fs = fs
        period = int(np.ceil(in_period))
        fs = int(period * freq)        
        data = resample_by_fs(indata, up=fs, down=old_fs)        
    else:
        data = indata
        period = int(in_period)        

    # if the kernel period is not matching the artifact period, 
    # filtering would be off
    kperiod, kwidth =  _estimate_prms_from_kernel(kernel)
    if kperiod != period:
        raise ValueError('Kernel is not matching artifact frequency. ' +
                         'Was the kernel correctly constructed?')

    #-------------------------------------------------------------------------
    fdata = np.convolve(data, kernel[::-1], 'same')
    #-------------------------------------------------------------------------
    if resample_flag:
        filtered = resample_by_count(fdata, in_samples)
        filtered = np.asanyarray(filtered)    
    else:
        filtered = fdata            
        
    return filtered
    
# %%
def _filter_2d(indata:ndarray, freq:int, fs:int, kernel:ndarray):        
    
    filtered = np.zeros(indata.shape)
    for idx, chandata in enumerate(indata):        
        filtered[idx,:] = _filter_1d(chandata, freq, fs, kernel)
    
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
        
    
    