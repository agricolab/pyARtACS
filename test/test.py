#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Test module for pyArtacs


'''
from artacs.kernel import create_kernel, _estimate_prms_from_kernel, filter_1d, filter_2d
import numpy as np
#%%
def test_kernel():
    freq = 10
    fs = 1000
    width = 5
    period = int(np.ceil(fs/freq))
    kernel = create_kernel(freq, fs, width)
    assert np.isclose(kernel.sum(), 0.0)
    assert kernel.shape[0] == 1001
    assert(_estimate_prms_from_kernel(kernel) == (period, width))
    
    kernel = create_kernel(freq, fs, width, 
                           left_mode='uniform',right_mode ='None')
    assert np.isclose(kernel.sum(), 0.0)
    assert kernel.shape[0] == 1001
    assert(_estimate_prms_from_kernel(kernel) == (period, width))
    assert np.all(np.isclose(kernel[::period],
                             np.array([-0.2, -0.2, -0.2, -0.2, -0.2,  1., 
                                       -0. , -0. , -0. , -0. , -0. ])))
    
    kernel = create_kernel(freq, fs, width, 
                           left_mode='gauss',right_mode ='exp')
    assert np.isclose(kernel.sum(), 0.0)
    assert kernel.shape[0] == 1001
    assert(_estimate_prms_from_kernel(kernel) == (period, width))
    assert np.all(np.isclose(kernel[::period],
                             np.array([-1.59837446e-05, -8.72682888e-04,
                                       -1.75283044e-02, -1.29517624e-01,
                                       -3.52065405e-01,  1.00000000e+00,
                                       -3.18204323e-01, -1.17060829e-01,
                                       -4.30642722e-02, -1.58424604e-02,
                                       -5.82811548e-03])))
    
    kernel = create_kernel(freq, fs, width, 
                           left_mode='linear',right_mode ='uniform')
    assert np.isclose(kernel.sum(), 0.0)
    assert kernel.shape[0] == 1001
    assert(_estimate_prms_from_kernel(kernel) == (period, width))
    assert np.all(np.isclose(kernel[::period],
                             np.array([-0.03333333, -0.06666667, -0.1,
                                       -0.13333333, -0.16666667, 1. , -0.1,
                                       -0.1, -0.1, -0.1,-0.1])))
    
    
    # test directionality of kernel
    fs = 1000
    freq = 10
    period = int(np.ceil(fs/freq))
    duration_in_s = 2
    t = np.linspace(1/fs, duration_in_s, num=fs*duration_in_s)
    data = np.zeros(np.shape(t))
    data[data.shape[0]//2] = 1
    kernel = create_kernel(freq, fs, 1, 
                           left_mode='uniform', right_mode ='none')
    filtered = filter_1d(data, fs, freq, kernel)
    assert np.where(filtered==1.0)[0][0] == 1000
    assert np.where(filtered==-1.0)[0][0] == 1100
    
    
    # test kernel removes sine perfectly
    fs = 1000
    freq = 10
    period = int(np.ceil(fs/freq))
    duration_in_s = 2
    t = np.linspace(1/fs, duration_in_s, num=fs*duration_in_s)
    data = np.sin(2*np.pi*freq*t)    
    kernel = create_kernel(freq, fs, 1, 
                           left_mode='uniform', right_mode ='none')
    filtered = filter_1d(data, fs, freq, kernel)
    assert np.all(np.isclose(filtered[period:], 0, 1e-10))

    # test kernel removes sine somewhat well when fs mismatch
    fs = 1000
    freq = 15
    for freq in range(10, 21, 1):        
        period = int(np.ceil(fs/freq))
        kernel = create_kernel(freq, fs, 1, 
                               left_mode='uniform', right_mode ='none')
        t = np.linspace(1/fs, duration_in_s, num=fs*duration_in_s)
        data = np.sin(2*np.pi*freq*t)    
        filtered = filter_1d(data, fs, freq, kernel)
        print('{:3.0f} '.format(freq),end='')
        print(np.max(np.abs(filtered[period*2:])))
        assert np.all(np.isclose(filtered[period*2:], 0, atol=1e-04))

    # test multi-channel filter
    fs = 1000
    freq = 10
    period = int(np.ceil(fs/freq))
    duration_in_s = 2
    t = np.linspace(1/fs, duration_in_s, num=fs*duration_in_s)
    data = np.sin(2*np.pi*freq*t)    
    data = np.vstack((data, data))
    kernel = create_kernel(freq, fs, 1, 
                               left_mode='uniform', right_mode ='none')
    filtered = filter_2d(data, fs, freq, kernel)
    for chan in filtered:
        assert np.all(np.isclose(chan[period:], 0, 1e-10))


    print('Test successful')
    
# %%
if __name__ == '__main__':
    test_kernel()
    
