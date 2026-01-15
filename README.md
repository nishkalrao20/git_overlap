# Comprehensive analysis of time-domain overlapping gravitational wave transients: A Lensing Study
Nishkal Rao <sup>1,2</sup>, Anuj Mishra <sup>3,2</sup>, Apratim Ganguly <sup>2</sup>, Anupreeta More <sup>2,4</sup>

<sub>1.Department of Physics, Indian Institute of Science Education and Research, Pashan, Pune - 411 008, India</sub>  
<sub>2.Inter-University Centre for Astronomy and Astrophysics (IUCAA), Post Bag 4, Ganeshkhind, Pune 411 007, India</sub>  
<sub>3.International Centre for Theoretical Sciences, Tata Institute of Fundamental Research, Bangalore 560089, India</sub>  
<sub>4.Kavli IPMU (WPI), UTIAS, The University of Tokyo, Kashiwa, Chiba 277-8583, Japan</sub>  

## Introduction
Next-generation gravitational-wave (GW) detectors will produce a high rate of temporally overlapping signals from unrelated compact binary coalescences. Such overlaps can bias parameter estimation (PE) and mimic signatures of other physical effects, such as gravitational lensing. In this work, we investigate how overlapping signals can be degenerate with gravitational lensing by focusing on two scenarios: Type-II strong lensing and microlensing by an isolated point-mass lens. We simulate quasicircular binary black-hole pairs with chirp-mass ratios $\mathscr{M}_{\mathrm{B}}/\mathscr{M}_{\mathrm{A}}\in\{0.5,\,1,\,2\}$, signal-to-noise ratio (SNR) ratios $\mathrm{SNR}_{\mathrm{B}}/\mathrm{SNR}_{\mathrm{A}}\in\{0.5,\,1\}$, and coalescence-time offsets $\Delta t_{\mathrm{c}}\in[-0.1,\,0.1]\,\mathrm{s}$, and extend to a population analysis. Bayesian PE and fitting-factor studies show that the Type-II lensing hypothesis is favored over the unlensed quasicircular hypothesis ($\log_{10}\mathscr{B}^{\mathrm{L}}_{\mathrm{U}}>1$) only in a small region of the overlapping parameter space with $\mathscr{M}_{\mathrm{B}}/\mathscr{M}_{\mathrm{A}}\gtrsim1$ and $|\Delta t_{\mathrm{c}}|\leq0.03\,\mathrm{s}$, with the inferred Morse index clustering near $n_j\simeq0.5$, indicative of Type-II lensing, for the cumulative study. Meanwhile, false evidence for microlensing signatures can arise because, to a reasonable approximation, the model produces two superimposed images whose time delay can closely match $|\Delta t_{\mathrm{c}}|$. The microlensing hypothesis is maximally favored ($\log_{10}\mathscr{B}^{\mathrm{L}}_{\mathrm{U}}\gg1$) for $\mathscr{M}_{\mathrm{B}}/\mathscr{M}_{\mathrm{A}}\gtrsim1$ and equal SNRs, increasing with $|\Delta t_{\mathrm{c}}|$. The inferred redshifted lens masses lie in the range $M_{\mathrm{L}}^z\sim10^2$--$10^{5}\,\mathrm{M}_{\odot}$ with impact parameters $y\sim0.1$--$3\,\mathrm{R_E}$. Overall, the inferred Bayes factor depends on relative chirp-mass ratios, relative loudness, difference in coalescence times, and also the absolute SNRs of the overlapping signals. Cumulatively, our results indicate that overlapping black-hole binaries with nearly equal chirp masses and comparable loudness are likely to be falsely identified as lensed. Such misidentifications are expected to become more common as detector sensitivities improve. While our study focuses on ground-based detectors using appropriate detectability thresholds, the findings naturally extend to next-generation GW observatories.

## Paper
[Phy. Rev. D](https://doi.org/10.1103/tflq-2xsd)
[arXiv:2510.17787](https://arxiv.org/abs/2510.17787)

### Code
The relevant overlapping waveform generation and fitting factor computation code can be found at `src/scripts/match_final/` containing [waveforms.py](https://github.com/nishkalrao20/git_overlap/blob/master/src/scripts/match_final/waveforms.py) for generating the time and frequency domain waveforms, and the Microlensed, Type II Strong lensed, and Eccentric waveforms. [FF_computation.py](https://github.com/nishkalrao20/git_overlap/blob/master/src/scripts/match_final/FF_computation.py) has the fitting factor module, whose usage can be learnt at [match.py](https://github.com/nishkalrao20/git_overlap/blob/master/src/scripts/match_final/match.py).

Packages required are [bilby](https://git.ligo.org/lscsoft/bilby), [pycbc](https://pycbc.org/), [gwmat](https://git.ligo.org/anuj.mishra/gwmat) and [gweat](https://gitlab.com/anuj-mishra/gweat), and their requirements. 

## License and Citation

![Creative Commons License](https://i.creativecommons.org/l/by-sa/3.0/us/88x31.png "Creative Commons License")

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 United States License](http://creativecommons.org/licenses/by-sa/3.0/us/).

We encourage use of these data in derivative works. If you use the material provided here, please cite the paper using the reference:

```
@article{Rao:2025poe,
    author = "Rao, Nishkal and Mishra, Anuj and Ganguly, Apratim and More, Anupreeta",
    title = "{Comprehensive analysis of time-domain overlapping gravitational wave transients: A Lensing Study}",
    eprint = "2510.17787",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P2500640",
    doi = "10.1103/tflq-2xsd",
    month = "10",
    year = "2025"
}
```
