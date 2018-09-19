#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:31:21 2018

@author: rgugg
"""
import numpy as np
import artacs.tools as tools
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
            data = tools.resample_by_fs(indata,
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
            outdata = tools.resample_by_count(outdata, 
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
            period_data, put_where = tools.signal2periods(fdata, 
                                                          period, offset=seed)
            component, score, l = tools.pca_largest(period_data)
            template = tools.comp2trace(component, score, kind='cubic')                
            amplitude = tools.estimate_artifact_peakiness(template,
                                                          fs, self.freq)      
            iteration += 1
            if amplitude < self.epsilon:
                converged = True                    
            else:
                fdata[put_where] -= template    
        return put_where, fdata[put_where]