# Python Implementation
Repo contains source code for creating and filtering EEG data from _periodic, non-sinusoidal_ and _non-stationary_ tCS artifacts using ___weighted comb filters___.

Includes also code for artifact removal using ___adaptive DFT___ and ___adaptive PCA___, and for simulation of tACS recordings.

This module is shared under a [X11 license](LICENSE).
Its development is supported by the [BMBF: FKZ 13GW0119](https://www.medizintechnologie.de/fileadmin/pdfs/Projektsteckbriefe_bekanntmachungen/IndiMedtech/13GW0119_Projektsteckbrief_NEU.pdf).

#### Example application
| Upper Limb Bipolar ECG recording <br> during 11 Hz tACS |<img src="./doc/source/_static/img/upper_limb_ecg.jpg" width = "400">|
|:----:|:----:|

#### Weighted Comb Filter
Artifacts can be _non-stationary_ and _non-sinusoidal_, but are required to be _periodic_. Comb filters natively support only frequencies which are integer divisibles of the sampling frequency. This can be circumvented by resampling the signal, and has been implemented.

##### Creation

The following example creates a kernel for a _classical_ causal comb filte for 
an artifcat with a period of 10Hz and sampled at 1000Hz:
```{python}        
    kernel = create_kernel(
        freq=10, fs=1000, width=1, left_mode="uniform", right_mode="none"
    )
```
    
or the superposition of moving averages (SMA) filter as discussed e.g. by [1], 
here for 5 periods at 10 Hz and at 1000Hz sampling rate:

```{python}    
    kernel = create_kernel(freq=10, fs:1000, width:5, 
                           left_mode:str='uniform', 
                           right_mode:str='uniform')
```    
    

[1]: Kohli, S., Casson, A.J., 2015. Removal of Transcranial a.c. Current Stimulation 
artifact from simultaneous EEG recordings by superposition of moving averages.
Conf Proc IEEE Eng Med Biol Soc 2015, 3436–3439. 
https://doi.org/10.1109/EMBC.2015.7319131

##### Application

The kernels, once created can be applied to the signal as follows:

```{python}    
    filtered_data = filter_2d(artifacted_data, freq=20, fs=1000, kernel:ndarray)
```
    
The kernel application is implemented for 2-dimensional data, which calls the 
1-dimensional implementation in a for-loop. Consider that you have to specify
the artifact frequency and sampling rate again. This is because the kernel only
makes sense if the period is an integer divisible of the sampling rate. 
If it is not, the signal is automaticall up-sampled, processed, and down-sampled.
This allows to remove the artifact, but is far from perfect. Additionally, 
the function estimates the period of the kernel, and throws an exception if 
it does not match the specified frequency. This ensures that the correct kernel
is used.

### Periodic Component Removal

An alternative is the creation and removal of periodic templates, until the 
artifact power is sufficiently suppressed. This can be achieved using 

```{python}
    from artacs import StepwiseRemover
    remover = StepwiseRemover(freq=20, s=1000)    
    filtered_data =  remover.process(artifacted_data)
```

See also [![DOI](https://zenodo.org/badge/87182503.svg)](https://zenodo.org/badge/latestdoi/87182503) for a similar implementation in Matlab.


