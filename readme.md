# Python Implementation
Repo contains source code for creating and filtering EEG data from _periodic, non-sinusoidal_ and _non-stationary_ tCS artifacts using ___weighted comb filters___.

Includes also code for artifact removal using ___adaptive DFT___ and ___adaptive PCA___, and for simulation of tACS recordings.

This module is shared under a [X11 license](LICENSE).
Its development is supported by the [BMBF: FKZ 13GW0119](https://www.medizintechnologie.de/fileadmin/pdfs/Projektsteckbriefe_bekanntmachungen/IndiMedtech/13GW0119_Projektsteckbrief_NEU.pdf).

#### Example application
| Upper Limb Bipolar ECG recording <br> during 11 Hz tACS |<img src="./doc/source/_static/img/upper_limb_ecg.jpg" width = "400">|
|:----:|:----:|
|__Recover the physiological signal__ <br>(which is ~120dB weaker than tACS) |<img src="./doc/source/_static/img/ecg_raw.png" width= "400">|

#### Weighted Comb Filter
Artifact can be _non-stationary_ and _non-sinusoidal_, but is required to be _periodic_. Comb filters natively support only frequencies which are integer divisibles of the sampling frequency. When artacs.kernel.run is used, the signal is automatically resampled, to circumvent this limitation. Note that the method still requires integer frequencies.

By default, the kernel is symmetric and weights are based empirically on the artifacts periodic autocorrelation.














See also [![DOI](https://zenodo.org/badge/87182503.svg)](https://zenodo.org/badge/latestdoi/87182503) for a similar implementation in Matlab.


