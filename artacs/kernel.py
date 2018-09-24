#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel based approaches

This module implements various periodic kernels inspired by comb filters. 
By using different weighting functions, e.g. uniform  for one or both sides 
of the kernel, the user can construct comb kernels and apply them on an artifacted
signal.

Creation
--------

The following example creates a kernel for a _classical_ causal comb filte for 
an artifcat with a period of 10Hz and sampled at 1000Hz::
    
    kernel = create_kernel(freq=10, fs:1000, width:1, 
                           left_mode:str='uniform', 
                           right_mode:str='none')
    
or the superposition of moving averages (SMA) filter as discussed e.g. by [1], 
here for 5 periods at 10 Hz and at 1000Hz sampling rate::
    
    kernel = create_kernel(freq=10, fs:1000, width:5, 
                           left_mode:str='uniform', 
                           right_mode:str='uniform')
    
    

[1]: Kohli, S., Casson, A.J., 2015. Removal of Transcranial a.c. Current Stimulation 
artifact from simultaneous EEG recordings by superposition of moving averages.
Conf Proc IEEE Eng Med Biol Soc 2015, 3436–3439. 
https://doi.org/10.1109/EMBC.2015.7319131

Application
-----------   

The kernels, once created can be applied to the signal as follows::
    
    filtered = filter_2d(indata, freq=20, fs=1000, kernel:ndarray)
    
The kernel application is implemented for 2-dimensional data, which calls the 
1-dimensional implementation in a for-loop. Consider that you have to specify
the artifact frequency and sampling rate again. This is because the kernel only
makes sense if the period is an integer divisible of the sampling rate. 
If it is not, the signal is automaticall up-sampled, processed, and down-sampled.
This allows to remove the artifact, but is far from perfect. Additionally, 
the function estimates the period of the kernel, and throws an exception if 
it does not match the specified frequency. This ensures that the correct kernel
is used.

Object-oriented implementation
------------------------------

There exists also an object-oriented implementation.


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
    '''create kernel from parameters
    
    args
    ----
    freq:int
        the frequency of the periodic artifact
    fs:int
        the sampling rate of the signal to be filtered
    width:int
        defines the number of periods in both directions
    left_mode:str {'uniform', 'none', 'gauss', 'linear', 'exp'}
        defines the shape of the left (causal) half of the kernel
    right_mode:str {'uniform', 'none', 'gauss', 'linear', 'exp'}
        defines the shape of the right half of the kernel

    returns
    -------
    kernel:ndarray
        the kernel for later application
        
    .. seealso::
    
       :func:`~.filter_1d`

    '''
    in_period = fs/freq    
    period = int(np.ceil(in_period))   
    if in_period != period:
        warn ('Only integer periods are natively supported.' + 
              'Will auto-resample to higher sampling rate')
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
def filter_1d(indata, fs:int, freq:int, kernel:ndarray):
    ''' filter a one-dimensional dataset with a predefined kernel

    args
    ----
    indata:ndarray
        one-dimensional artifacted signal 
    freq:int
        the frequency of the periodic artifact
    fs:int
        the sampling rate of the signal to be filtered
    kernel:ndarray
        the kernel for later application, see also :func:`~.create_kernel`
        
        
    returns
    -------
    filtered:ndarray
        one-dimensional signal with artifact removed
        
    .. seealso::
    
       :func:`~.filter_2d`
    ''' 
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
def apply_kernel(indata:ndarray, fs:int, freq:int, kernel:ndarray):        
    ''' filter a two-dimensional dataset with a predefined kernel

    args
    ----
    indata:ndarray
        two-dimensional artifacted signal, dimensions are channel x samples
    freq:int
        the frequency of the periodic artifact
    fs:int
        the sampling rate of the signal to be filtered
    kernel:ndarray
        the kernel for later application, see also :func:`~.create_kernel`
        
        
    returns
    -------
    filtered:ndarray
        two-dimensional signal with artifact removed
        
    .. seealso::
    
       :func:`~.filter_1d`
    ''' 
    filtered = np.zeros(indata.shape)
    for idx, chandata in enumerate(indata):        
        filtered[idx,:] = filter_1d(chandata, fs, freq, kernel)
    
    return filtered

#%%
class CombKernel():
    
    def __init__(self, freq:int, fs:int, width:int, 
                  left_mode:str='uniform', 
                  right_mode:str='uniform') -> None:
        
        self._freq = freq
        self._fs = fs
        self._width = width
        self._left_mode = left_mode
        self._right_mode = right_mode
        self._update_kernel()

    def _update_kernel(self):
        self._kernel = create_kernel(self._freq, self._fs, 
                                     self._width, 
                                     self._left_mode, 
                                     self._right_mode)      
       
    def __call__(self, indata:np.array) -> ndarray:
        return apply_kernel(indata=indata, freq=self._freq, fs=self._fs, 
                         kernel=self._kernel)
    
    def __repr__(self):
        return (f'KernelFilter({self._freq}, {self._fs}, {self._width}, ' +
               f"'{self._left_mode}', '{self._right_mode}')")
    
