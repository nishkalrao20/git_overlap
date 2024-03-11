# Inject BBH Injections (SINGLES/ PAIRS) in Detectors 

import pycbc
import pickle
import numpy as np
import pandas as pd
import pycbc.psd
import pycbc.waveform
import pycbc.noise.reproduceable
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'text.usetex' : True})

# Importing the injection parameters

waveform_metadata_a, waveform_metadata_b = pickle.load(open('git_overlap/src/output/overlap_injection/Waveform A Meta Data.pkl', 'rb')), pickle.load(open('git_overlap/src/output/overlap_injection/Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
for key,val in waveform_metadata_a['H1']['parameters'].items():
    exec(key +'_a' + '=val')
for key,val in waveform_metadata_b['H1']['parameters'].items():
    exec(key +'_b' + '=val')

delta_f = 1
minimum_frequency = 20
maximum_frequency = 1024
sampling_frequency = 2048
delta_t = 1/sampling_frequency
duration = 16
tsamples = int(duration / delta_t)

def gen_wf(mass_1, mass_2, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time):
    """Generate time domain waveforms for the injection parameters"""

    hp, hc = pycbc.waveform.get_td_waveform(approximant='IMRPhenomPv2', mass1=mass_1, mass2=mass_2, distance=luminosity_distance, spin1z=spin1z, spin2z=spin2z, 
                                            inclination=incl, coa_phase=phase, delta_t = delta_t, f_lower=minimum_frequency, f_final=maximum_frequency)    

    hp._epoch, hc._epoch = hp._epoch+geocent_time, hc._epoch+geocent_time   # Setting the start times

    return hp, hc

def inject_wf(mass_1, mass_2, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time):
    """ Returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""
    
    ht = dict()
    for i, det in enumerate(['H1', 'L1', 'V1']):

        det_obj = pycbc.detector.Detector(det)   # Loading the detector

        hp, hc = gen_wf(mass_1, mass_2, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time)   # Generate the waveform

        fp, fc = det_obj.antenna_pattern(ra, dec, psi, geocent_time)   # Generate the antenna beam pattern functions for this particular sky localization
        h = hp*fp + hc*fc   # Projected timeseries

        ht[det] = det_obj.project_wave(hp, hc, ra, dec, psi)   # Calculate the waveform projected into each detector

    return ht

def read_psd(det):
    """Reading the PSD files for the detectors"""

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

def inject_wf_noise(mass_1, mass_2, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time):
    """ Returns injection projections of a signal with colored Gaussian noise onto the Hanford, Livingston, Virgo detectors"""

    ht, strain = dict(), dict()
    for i, det in enumerate(['H1', 'L1', 'V1']):

        det_obj = pycbc.detector.Detector(det)   # Loading the detector

        hp, hc = gen_wf(mass_1, mass_2, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time)   # Generate the waveform

        ht[det] = det_obj.project_wave(hp, hc, ra, dec, psi)   # Calculate the waveform projected into each detector

        nt = pycbc.noise.reproduceable.colored_noise(read_psd(det), geocent_time - duration/2,  geocent_time + duration/2, seed = np.random.randint(low=0, high=2**32-1), sample_rate = sampling_frequency, low_frequency_cutoff = minimum_frequency)

        strain[det] = nt.inject(ht[det], copy=True)

    return ht, strain

# Generating & Injecting the waveforms

ht_a, strain_a = inject_wf_noise(mass_1_a, mass_2_a, a_1_a, a_2_a, luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a)
ht_b, strain_b = inject_wf_noise(mass_1_b, mass_2_b, a_1_b, a_2_b, luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b)

# Plotting the waveforms

fig, ax = plt.subplots(2,3, figsize=(18, 10))
fig.suptitle('\\textbf{WAVEFORM INJECTIONS: SINGLES}')
ax[0,0].plot(ht_a['H1'].sample_times, ht_a['H1'], label='$h(t)$')
ax[0,0].set_title('H1 - SINGLES A')
ax[0,0].set_xlabel('Time $t$')
ax[0,0].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
ax[0,1].plot(ht_a['L1'].sample_times, ht_a['L1'], label='$h(t)$')
ax[0,1].set_title('L1 - SINGLES A')
ax[0,1].set_xlabel('Time $t$')
ax[0,1].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
ax[0,2].plot(ht_a['V1'].sample_times, ht_a['V1'], label='$h(t)$')
ax[0,2].set_title('V1 - SINGLES A')
ax[0,2].set_xlabel('Time $t$')
ax[0,2].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
ax[1,0].plot(ht_b['H1'].sample_times, ht_b['H1'], label='$h(t)$')
ax[1,0].set_title('H1 - SINGLES B')
ax[1,0].set_xlabel('Time $t$')
ax[1,0].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
ax[1,1].plot(ht_b['L1'].sample_times, ht_b['L1'], label='$h(t)$')
ax[1,1].set_title('L1 - SINGLES B')
ax[1,1].set_xlabel('Time $t$')
ax[1,1].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
ax[1,2].plot(ht_b['V1'].sample_times, ht_b['V1'], label='$h(t)$')
ax[1,2].set_title('V1 - SINGLES B')
ax[1,2].set_xlabel('Time $t$')
ax[1,2].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
plt.savefig('git_overlap/src/output/overlap_injection/INJECTIONS: SINGLES.png')
plt.close()