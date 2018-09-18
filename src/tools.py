#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
another advantage of using scipy.linalg over numpy.linalg is that it is always
compiled with BLAS/LAPACK support, while for numpy this is optional. Therefore,
the scipy version might be faster depending on how numpy was installed.
'''
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import numpy as np
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
