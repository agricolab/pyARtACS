#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:01:06 2018

@author: rgugg
"""
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from tools import pca_largest
import matplotlib.pyplot as plt
# %%
def modify_kernel(kernel, mode='uniform'):
    width = (kernel[:kernel.shape[0]//2]<0).sum()
    period = int(np.unique(np.diff(np.where(kernel!=0)[0])))
    if mode=='uniform':
        return kernel
    elif mode == 'exponential':
        weights = signal.exponential(((width-1)*2)+1)[:width]
        
        
    weights /= weights.sum() *width       
    weights = np.hstack((weights, 1.0, weights[::-1]))
        
    
    
    kernel[::period] * weights
    
def direct_kernel(kernel, direction='symmetric'):
    kernel = kernel.copy()
    if direction == 'causal' or direction == 'left':
        kernel[kernel.shape[0]//2+1:] = 0
        kernel[:kernel.shape[0]//2-1] *= 2
        return kernel
    elif direction == 'right':
        kernel[kernel.shape[0]//2+1:] *= 2
        kernel[:kernel.shape[0]//2-1] = 0
        return kernel        
    elif direction == 'symmetric' or direction == 'sym':
        return kernel    
    else:
        raise NotImplementedError('Direction unknown')
    
def create_uniform_symmetric_kernel(freq, fs, width):
    '''Create a uniform comb kernel
    '''
    in_period = fs/freq    
    period = int(np.ceil(in_period))
   
    k = np.zeros((period*width*2)+1)
    k[0::period] = -1/(width*2)
    k[width*period] = 1
    return k
    
def _test_kernel():
    freq = 12
    fs = 1000
    width = 5
    period = int(np.ceil(fs/freq))
    base_kernel = create_uniform_symmetric_kernel(freq, fs, width)
    assert base_kernel.sum() == 0
    assert base_kernel.shape[0] == 841
    
    kernel = direct_kernel(base_kernel, 'symmetric')
    assert kernel.sum() == 0    
    assert kernel.shape[0] == 841
    assert (kernel[::period] - np.array([-0.1, -0.1, -0.1, -0.1, -0.1,  1. , 
                                    -0.1, -0.1, -0.1, -0.1, -0.1])).sum() == 0
    
    kernel = direct_kernel(base_kernel, 'causal')
    assert kernel.sum() == 0    
    assert kernel.shape[0] == 841
    assert (kernel[::period] - np.array([-0.2, -0.2, -0.2, -0.2, -0.2,  1. ,  
            0. ,  0. ,  0. ,  0. ,  0. ])).sum() == 0
    
    kernel = direct_kernel(base_kernel, 'right')
    assert kernel.sum() == 0    
    assert kernel.shape[0] == 841
    assert (kernel[::period] - np.array([ 0. ,  0. ,  0. ,  0. ,  0. ,  1. , 
            -0.2, -0.2, -0.2, -0.2, -0.2])).sum() == 0 
    
    kernel =modify_kernel(base_kernel, 'uniform')
    assert (kernel-base_kernel).sum()==0
    
    kernel =modify_kernel(base_kernel, 'exp')
    
# %%
def kernel_filter(indata, freq, fs, width):        
    
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

# %%    
class StepwiseRemover():
    
    def __init__(self, fs=1000, freq=None, period_steps=2, 
                 epsilon=0.01, max_iterations=10):
        self.true_fs = fs
        self.freq = freq
        if freq is not None:
            self.true_period = fs/freq
            self.resample_flag = (self.true_period != int(fs/freq))
        else:
            self.true_period = None            
                
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.period_steps = period_steps 
    
    def calc_seeds(self, period):
        'derive seedpoints for starting the cutting into periods'
        seeds = np.unique(np.linspace(0, period, self.period_steps+1, 
                                      dtype='int32'))
        seeds = seeds[seeds!=period] 
        return seeds
    
    def inbound_resample(self, indata):
        'resample so that (artifact_period* artifact_frequency) is an integer'
        if self.resample_flag:                   
            period = int(np.ceil(self.true_period))
            fs = int(period * self.freq)            
            data = resample_by_fs(indata,
                                   up=fs,
                                   down=self.true_fs, 
                                   axis=0)       
            self.sample_count = indata.shape[0]
        else:
            data = indata
            fs = self.fs
            period = int(self.true_period)                
        return data, period, fs

    def outbound_resample(self, outdata, fs):
        'reverse an earlier resampling, if it was necessary'
        if self.resample_flag:                    
            outdata = resample_by_count(outdata, 
                                        self.sample_count,
                                        axis=0)         
        return outdata

    def prepare_data(self, indata):
        'resample and derive seedpoints'        
        valid_data = indata[np.invert(np.isnan(indata))]
        data, period, fs = self.inbound_resample(valid_data)
        seeds = self.calc_seeds(period)
        return data, period, fs, seeds        
    
    def process(self, indata):
        'process all channels of a dataset'
        if self.true_period is None:
            print('Invalid period length, skipping artifact removal')
            return indata
        
        num_channels, num_samples = indata.shape
        outdata = np.empty((indata.shape))
        outdata.fill(np.nan)
        print('[',end='')
        for chan_idx, chan_data in enumerate(indata):            
            outdata[chan_idx,:] = self.process_channel(chan_data)
            print('.',end='')
        print(']',end='\n')
        return outdata
    
    def process_channel(self, indata):    
        'process a single channels of data'
        if self.true_period is None:
            print('Invalid period length, skipping artifact removal')
            return indata
        
        data, period, fs, seeds = self.prepare_data(indata)     
        outdata = np.empty((data.shape[0], seeds.shape[0]+1))
        outdata.fill(np.nan)
        
        for seed_idx, seed in enumerate(seeds):         
            idx, fdata = self._process(data, period, fs, seed)
            outdata[idx, seed_idx] = fdata            

        missing_part = idx[-1]+period != outdata.shape[0]
        if missing_part: #perform filtering in time-reversed data
            idx, fdata = self._process(data[::-1], period, fs, seed=0)
            outdata[outdata.shape[0]-idx[-1]-1:, -1] = fdata[::-1]
            
        outdata = np.nanmean(outdata, axis=1)
        outdata = self.outbound_resample(outdata, fs)                
        
        return outdata

    def _process(self, data, period, fs, seed):
        converged = False
        iteration = 0
        fdata = data.copy()
        while not converged:        
            period_data, put_where = signal2periods(fdata, period, offset=seed)
            component, score, l = pca_largest(period_data)
            template = comp2trace(component, score, kind='cubic')                
            amplitude = estimate_artifact_peakiness(template, fs, self.freq)      
            iteration += 1
            if amplitude < self.epsilon:
                converged = True                    
            else:
                fdata[put_where] -= template    
        return put_where, fdata[put_where]
        
def comp2trace(component, score, kind='cubic'):
    'transfrom periodic component scores into a continuous signal'
    block = np.repeat(score, component.shape[0], axis = 1).T
    trace = np.reshape(block,(np.prod(block.shape),))
    halfperiod = score.shape[0]/2
    x = np.linspace(halfperiod, trace.shape[0]-halfperiod, 
                    num=component.shape[0]-1, endpoint=True)        
    calcperiod = np.ceil(score.shape[0]/2)
    xi = np.arange(calcperiod, trace.shape[0]-calcperiod)    
    amplitude = interp1d(x, component[:-1].T, kind=kind)
    weights = amplitude(xi)
    weights = np.pad(weights[0,:], (int(calcperiod),int(calcperiod)), 'edge')     
    template = (weights*trace)
    return template

def signal2periods(xdata, period, offset=0):
    'cut a continuous signal into periods, starting a a specific offset'
    samples = xdata.shape[0]
    timepoints = np.arange(offset, samples-period, period)
    put_where = np.arange(timepoints[0],timepoints[-1])
    period_data = np.zeros((period,timepoints.shape[0]-1))
    for sidx, (a,b) in enumerate(zip(timepoints[0:-1],timepoints[1:])):                  
        period_data[:,sidx] = xdata[a:b] 
    return period_data, put_where
 
def estimate_artifact_peakiness(signal_trace, fs, freq):    
    f, pxx = signal.welch(signal_trace, fs=fs, scaling='density', nperseg=fs)
    decibel = np.log10(pxx)
    foi = f[::freq] 
    idx = [_f in foi for _f in f]    
    signal_amplitude = np.max(decibel[np.invert(idx)])
    artifact_amplitude = np.max(decibel[idx])
    amplitude = artifact_amplitude - signal_amplitude
    return amplitude
    
def resample_by_fs(indata, up=1000, down=1000, axis=0):
    l = indata.shape[0]
    new_sample_count = int(np.ceil(l*(up/down)))
    sample_idx = np.atleast_1d(np.linspace(0,l, l))
    f = interp1d(np.atleast_1d(sample_idx),np.atleast_1d(indata), kind='cubic')
    new_sample_idx = np.linspace(0, l, new_sample_count)
    return f(new_sample_idx)

def resample_by_count(indata, new_sample_count, axis=0):
    l = indata.shape[0]
    sample_idx = np.atleast_1d(np.linspace(0,l, l))
    f = interp1d(np.atleast_1d(sample_idx),np.atleast_1d(indata), kind='cubic')
    new_sample_idx = np.linspace(0, l, new_sample_count)
    return f(new_sample_idx)
#%%
if __name__ == '__main__':
    # %%
    freq = 10
    fs = 1000
    indata = np.atleast_2d(np.sin(2*np.pi*freq*np.arange(0,2.01,1/fs)))    
    width = 4
    plt.cla()
    plt.plot(indata[0,:])
    #plt.plot(pdata)
    plt.plot(kernel_filter(indata, freq, fs, width)[0,:])