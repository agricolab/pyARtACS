#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Test module for pyArtacs


'''
from artacs.kernel import create_kernel
import numpy as np
#%%
def test_kernel():
    freq = 10
    fs = 1000
    width = 5
    period = int(np.ceil(fs/freq))
    kernel = create_kernel(freq, fs, width)
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    
    kernel = create_kernel(freq, fs, width, 
                           left_mode='uniform',right_mode ='None')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert np.all(np.isclose(kernel[::period],
                             np.array([-0.2, -0.2, -0.2, -0.2, -0.2,  1., 
                                       -0. , -0. , -0. , -0. , -0. ])))
    
    kernel = create_kernel(freq, fs, width, 
                           left_mode='gauss',right_mode ='exp')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert np.all(np.isclose(kernel[::period],
                             np.array([-1.59837446e-05, -8.72682888e-04,
                                       -1.75283044e-02, -1.29517624e-01,
                                       -3.52065405e-01,  1.00000000e+00,
                                       -3.18204323e-01, -1.17060829e-01,
                                       -4.30642722e-02, -1.58424604e-02,
                                       -5.82811548e-03])))
    
    kernel = create_kernel(freq, fs, width, 
                           left_mode='linear',right_mode ='uniform')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert np.all(np.isclose(kernel[::period],
                             np.array([-0.  , -0.05, -0.1 , -0.15, -0.2 ,  1., 
                                       -0.1 , -0.1 , -0.1 , -0.1 , -0.1 ])))
    
    print('Test successful')
    
# %%
if __name__ == '__main__':
    test_kernel()
    