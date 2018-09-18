#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:03:00 2017

@author: Robert Guggenberger
"""

import scipy.signal as sig
from helper.peaks import detect_peaks as _detect_peaks
import wyrm as _wyrm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as _plt
import sklearn.decomposition as sk
# %%


def plotonsubs(*args):
    count = len(args)
    fig, ax = _plt.subplots(1, count)
    for _ax, item in zip(ax, args):
        _ax.plot(item, 'k')
        _ax.plot(np.mean(item, 1), 'r')


def phaseplot(timeseries):
    h = sig.hilbert(timeseries)
    r = np.real(h)
    i = np.imag(h)
    t = range(0, len(timeseries), 1)
    _plt.figure()
    ax = _plt.axes(projection='3d')
    ax.plot(t, r, i, '-b')
    return ax


def pcaplot(blocks):
    pca = sk.PCA(n_components=2)
    pca.fit(blocks)
    S = pca.fit_transform(blocks)
    t = range(0, len(blocks), 1)
    _plt.figure()
    ax = _plt.axes(projection='3d')
    ax.plot(t, S[:, 0], S[:, 1], '-b')
    return ax
# %%


def filtfilt(data, f_low=1, f_high=25, fs=None, butter_ord=4):
    if fs is None:
        if hasattr(data, 'fs'):
            fs = data.fs
        else:
            fs = 500

    fn = fs / 2
    b, a = sig.butter(butter_ord, [f_low / fn, f_high / fn],
                      btype='band')
    return _wyrm.processing.filtfilt(data, b, a, timeaxis=-2)


def find_qrs(data, window=(500, 500), mpd=750):
    peakind = _detect_peaks(-data.data[:, 0], mpd=mpd)
    ecg = []
    arm = []
    for idx in peakind:
        sel = range(idx - window[0], idx + window[1], 1)
        if sel[0] > 0 and sel[-1] < len(data.data):
            ecg.append(-data.data[sel, 0])
            arm.append(data.data[sel, 2] - data.data[sel, 1])
    return np.transpose(np.asanyarray(ecg)), np.transpose(np.asanyarray(arm))
