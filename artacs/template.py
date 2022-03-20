#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Template based artifact removal

Algorithm description
---------------------

:class:`~.StepwiseRemover` takes the period of the tACS artifact, and splits the timeseries into segments of that length, if necessary after resampling, to have an integer length period. It calculates the strongest principal components of this matrix (segments x period_count) and removes it until the residual power at the frequency of the tACS artifact in relation to neighbouring frequencies is below a certain threshold. To prevent discrete steps at the boundaries of the segments, this algorithm is repeated for different initial starting points. see also [1]

[1]: Guggenberger, R., & Gharabaghi, A. (2021). Comb filters for the removal of transcranial current stimulation artifacts from single channel EEG recordings. Current Directions in Biomedical Engineering, 7(2), 383–386. https://doi.org/10.1515/cdbme-2021-2097


Application
-----------   

Example::

    remover = StepwiseRemover(fs=1000, freq=50)
    remover.process(data)


"""
from numpy import ndarray
import numpy as np
import artacs.tools as tools
import logging

logger = logging.Logger(__name__)
# %%
class StepwiseRemover:
    """ Stepwise removal of tACS artifacts     
    
    args
    ----
    fs: int 
        sampling frequency of the data
    freq: float
        frequency of the tACS artifact
    period_steps: int
        across how many periods the seed points are selected
    epsilon: float
        the threshold for the residual power at the frequency of the tACS artifact
    max_iterations: int
        how many iterations are performed
    verbose: bool
        sets the verbosity of the procedure
        

    
    """

    def __init__(
        self,
        fs=1000,
        freq=None,
        period_steps=2,
        epsilon=0.01,
        max_iterations=10,
        verbose=True,
    ):

        self.verbose = verbose
        self.true_fs = fs
        self.freq = freq
        if freq is not None:
            self.true_period = fs / freq
            self.resample_flag = self.true_period != int(fs / freq)
        else:
            self.true_period = None

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.period_steps = period_steps

    def calc_seeds(self, period):
        "derive seedpoints for starting the cutting into periods"
        seeds = np.unique(
            np.linspace(0, period, self.period_steps + 1, dtype="int32")
        )
        seeds = seeds[seeds != period]
        return seeds

    def inbound_resample(self, indata):
        "resample so that (artifact_period* artifact_frequency) is an integer"
        self.sample_count = indata.shape[0]
        if self.resample_flag:
            period = int(np.ceil(self.true_period))
            fs = int(np.ceil(period * self.freq))
            data = tools.resample_by_fs(
                indata, up=fs, down=self.true_fs, axis=0
            )

        else:
            data = indata
            fs = self.true_fs
            period = int(self.true_period)
        return data, period, fs

    def outbound_resample(self, outdata, fs):
        "reverse an earlier resampling, if it was necessary"
        if self.resample_flag:
            outdata = tools.resample_by_count(
                outdata, self.sample_count, axis=0
            )
        return outdata

    def prepare_data(self, indata):
        "resample and derive seedpoints"
        valid_data = indata[np.invert(np.isnan(indata))]
        data, period, fs = self.inbound_resample(valid_data)
        seeds = self.calc_seeds(period)
        return data, period, fs, seeds

    def __call__(self, indata: ndarray) -> ndarray:
        return self.process(indata)

    def process(self, indata: ndarray):
        """process all channels of a dataset
        
        args
        ----
        indata: ndarray
            the two-dimensional data to be processed (channels x samples)
        
        returns
        -------
        outdata: ndarray
            the filtered data

        """
        if self.true_period is None:
            print("Invalid period length, skipping artifact removal")
            return indata

        if len(indata.shape) == 1:
            num_channels, num_samples = 1, indata.shape[0]
            indata = np.atleast_2d(indata)
        elif len(indata.shape) == 2:
            num_channels, num_samples = indata.shape
        else:
            raise ValueError("Unspecified dimensionality of the dataset")
        outdata = np.empty((indata.shape))
        outdata.fill(np.nan)
        if self.verbose:
            print("[", end="")
        for chan_idx, chan_data in enumerate(indata):
            outdata[chan_idx, :] = self.process_channel(chan_data)
            if self.verbose:
                print(".", end="")
        if self.verbose:
            print("]", end="\n")
        return np.squeeze(outdata)

    def process_channel(self, indata: ndarray) -> ndarray:
        "process a single channels of data"
        if self.true_period is None:
            print("Invalid period length, skipping artifact removal")
            return indata

        data, period, fs, seeds = self.prepare_data(indata)
        outdata = np.empty((data.shape[0], seeds.shape[0] + 1))
        outdata.fill(np.nan)

        for seed_idx, seed in enumerate(seeds):
            idx, fdata = self._process(data, period, fs, seed)
            outdata[idx, seed_idx] = fdata

        missing_part = idx[-1] + period != outdata.shape[0]
        if missing_part:  # perform filtering in time-reversed data
            idx, fdata = self._process(data[::-1], period, fs, seed=0)
            outdata[outdata.shape[0] - idx[-1] - 1 :, -1] = fdata[::-1]

        outdata = np.nanmean(outdata, axis=1)
        outdata = self.outbound_resample(outdata, fs)

        return outdata

    def _process(self, data: ndarray, period: int, fs: int, seed: int):
        converged = False
        iteration = 0
        fdata = data.copy()
        while not converged:
            period_data, put_where = tools.signal2periods(
                fdata, period, offset=seed
            )
            component, score, l = tools.pca_largest(period_data)
            template = tools.comp2trace(component, score, kind="cubic")
            amplitude = tools.estimate_artifact_peakiness(
                template, fs, self.freq
            )
            iteration += 1
            if amplitude < self.epsilon:
                converged = True
            else:
                fdata[put_where] -= template
        return put_where, fdata[put_where]
