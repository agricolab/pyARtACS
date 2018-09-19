#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
another advantage of using scipy.linalg over numpy.linalg is that it is always
compiled with BLAS/LAPACK support, while for numpy this is optional. Therefore,
the scipy version might be faster depending on how numpy was installed.
'''
import numpy as np
from scipy.signal import welch
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
#%%
def pca(data):
    data -= data.mean(axis=0)      
    R = np.cov(data, rowvar=False)
    
    eigen_value, eigen_vector = eigh(R)        
        
    idx = np.argsort(eigen_value)[::-1]    
    eigen_vector = eigen_vector[:,idx]    
    eigen_value = eigen_value[idx]

    eigen_value /= eigen_value.sum(axis=0)
    score = np.dot(eigen_vector.T, data.T).T     
    
    return eigen_vector, score, eigen_value

def pca_largest(data):
    data -= data.mean(axis=0)      
    R = np.cov(data, rowvar=False)
    eigen_value, eigen_vector = eigsh(R, 1, which='LM')            
    score = np.dot(eigen_vector.T, data.T).T     
    #return eigen_vector[:,0], score[:,0], eigen_value
    return eigen_vector, score, eigen_value


def pca_reduce(x, dimensions=1):
    x = np.asarray(x)
    
    if dimensions==1:
        c,s,l = pca_largest(x.T)
        c = c[:,0]
        s = s[:,0]
        if c[1]<0:
            c = -c
            s = -s
        return s, c
    else:
    
        pca = decomposition.PCA()
        x_std = StandardScaler().fit_transform(x)
        pca.fit_transform(x_std)
        coeffs = pca.components_[:,:dimensions].copy()
        scores = pca.fit_transform(x.T)
        return scores[:,:dimensions],coeffs
 
# %%
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

# %%
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
    f, pxx = welch(signal_trace, fs=fs, scaling='density', nperseg=fs)
    decibel = np.log10(pxx)
    foi = f[::freq] 
    idx = [_f in foi for _f in f]    
    signal_amplitude = np.max(decibel[np.invert(idx)])
    artifact_amplitude = np.max(decibel[idx])
    amplitude = artifact_amplitude - signal_amplitude
    return amplitude