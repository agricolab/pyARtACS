#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Test module for pyArtacs


'''
from artacs.kernel import _modify_kernel_direction, _create_uniform_symmetric_kernel, _modify_kernel_mode
import numpy as np
#%%
def test_kernel():
    freq = 10
    fs = 1000
    width = 5
    period = int(np.ceil(fs/freq))
    base_kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    assert base_kernel.sum() == 0
    assert base_kernel.shape[0] == 1001
    
    kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    kernel = _modify_kernel_direction(kernel, 'symmetric')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert (kernel[::period] - np.array([-0.1, -0.1, -0.1, -0.1, -0.1,  1. , 
                                    -0.1, -0.1, -0.1, -0.1, -0.1])).sum() == 0
    
    kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    kernel = _modify_kernel_direction(kernel, 'causal')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert (kernel[::period] - np.array([-0.2, -0.2, -0.2, -0.2, -0.2,  1. ,  
            0. ,  0. ,  0. ,  0. ,  0. ])).sum() == 0
    
    kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    kernel = _modify_kernel_direction(kernel, 'right')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert (kernel[::period] - np.array([ 0. ,  0. ,  0. ,  0. ,  0. ,  1. , 
            -0.2, -0.2, -0.2, -0.2, -0.2])).sum() == 0 
    
    kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    kernel = _modify_kernel_mode(kernel, 'uniform')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert (kernel-base_kernel).sum()==0
    
    kernel = _create_uniform_symmetric_kernel(freq, fs, width)
    kernel = _modify_kernel_mode(kernel, 'exp')
    assert kernel.sum() == 0
    assert kernel.shape[0] == 1001
    assert np.all(np.isclose(kernel[::period], 
                 np.array([-0.00582812, -0.01584246, -0.04306427, -0.11706083, 
                           -0.31820432,  1.        , -0.31820432, -0.11706083, 
                           -0.04306427, -0.01584246, -0.00582812])))
    
    print('Test successful')
    
# %%
if __name__ == '__main__':
    test_kernel()
    