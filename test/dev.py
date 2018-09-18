#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:07:58 2017

@author: rgugg
"""
# %%
import sys
sys.path.append('/media/rgugg/storage/projects/globalProjects/pyArtACS/src')
import helper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wyrm.io
import numpy as np
import scipy as sp
import sklearn.decomposition as sk
# %%
eeg = helper.load_brain_vision_data(
    '/media/rgugg/storage/projects/globalProjects/pyArtACS/data/10Hz3ma.vhdr')
eeg = helper.filtfilt(eeg, f_low=1, f_high=25)
ecg, arm = helper.find_qrs(eeg, window=(400, 400), mpd=800)
helper.plotonsubs(ecg, arm)

# %%
eeg = wyrm.io.load_brain_vision_data(
    '/media/rgugg/storage/projects/globalProjects/pyArtACS/data/clean.vhdr')
eeg = helper.filtfilt(eeg, f_low=1, f_high=25)
ecg, arm = helper.find_qrs(eeg, window=(400, 400), mpd=800)
helper.plotonsubs(ecg, arm)
plt.show()

# %%
cntArm = (eeg.data[:, 2] - eeg.data[:, 1])
window = range(1000, 4000, 1)
helper.phaseplot(cntArm[window])
helper.pcaplot(np.reshape(cntArm[window], (1000, 3)))
# %%
pca = sk.PCA()
pca.fit(arm)
S = pca.fit_transform(np.reshape(cntArm[window], (750, 4)))
fig, ax = plt.subplots(2, 3)
for idx in range(0, 6, 1):
    ax.flatten()[idx].plot(S[:, idx])
# %%
window = range(1000, 11000, 1)
sig = np.transpose(np.reshape(cntArm[window], (100, 100)))
helper.pcaplot(sig)
# %%
pca = sk.PCA()
pca.fit(arm)
S = pca.fit_transform(sig)
fig, ax = plt.subplots(25, 4)
for idx in range(0, 100, 1):
    ax.flatten()[idx].plot(S[:, idx])

plt.show()
