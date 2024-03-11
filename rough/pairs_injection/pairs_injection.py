# Simulate BBH PAIRS Injections in Detectors

import pycbc
import pickle
import pycbc.psd
import pycbc.waveform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'text.usetex' : True})

waveform_metadata_a, waveform_metadata_b = pickle.load(open('git_overlap/src/output/overlap_injection/Waveform A Meta Data.pkl', 'rb')), pickle.load(open('git_overlap/src/output/overlap_injection/Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
for key,val in waveform_metadata_a['H1']['parameters'].items():
    exec(key +'_a' + '=val')
for key,val in waveform_metadata_b['H1']['parameters'].items():
    exec(key +'_b' + '=val')

delta_f = 1
minimum_frequency = 20
maximum_frequency = 1024
sampling_frequency = 2048

# Generating the SINGLES waveforms

hp_a, hc_a = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=mass_1_a, mass2=mass_2_a, distance=luminosity_distance_a, spin1x=a_1_a, spin2x=a_2_a, 
                                            inclination=incl_a, coa_phase=phase_a, delta_f=delta_f, f_lower=minimum_frequency, f_final=maximum_frequency)    
hp_b, hc_b = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=mass_1_b, mass2=mass_2_b, distance=luminosity_distance_b, spin1x=a_1_b, spin2x=a_2_b, 
                                            inclination=incl_b, coa_phase=phase_b, delta_f=delta_f, f_lower=minimum_frequency, f_final=maximum_frequency)    

delta_b = np.random.uniform(0,0.5)   # Strong Bias

# Injecting the polarizations
 
det_obj_list, det_list = [pycbc.detector.Detector('H1'), pycbc.detector.Detector('L1'), pycbc.detector.Detector('V1')], ['H1', 'L1', 'V1']    # Setting the detectors

for i, det_obj in enumerate(det_obj_list):

    Fp_a, Fc_a = det_obj.antenna_pattern(ra_a, dec_a, psi_a, geocent_time_a)   # Antenna Patterns
    h_a = Fp_a*hp_a + Fc_a*hc_a   # h = F+*h+ + Fx*hx

    Fp_b, Fc_b = det_obj.antenna_pattern(ra_b, dec_b, psi_b, geocent_time_b)   # Antenna Patterns
    h_b = Fp_b*hp_b + Fc_b*hc_b   # h = F+*h+ + Fx*hx

    # Converting the waveforms to time domain

    ht_a = h_a.to_timeseries(delta_t=h_a.delta_t)
    ht_b = h_b.to_timeseries(delta_t=h_b.delta_t)

    # Padding the arrays to equalize the length

    ht_b = np.lib.pad(ht_b,(int(delta_b*sampling_frequency),0))
    ht_a = np.lib.pad(ht_a,(0,len(ht_b)-len(ht_a)))

    ht = np.add(ht_a,ht_b)   # Adding the waveforms

    # Generating the polarizations in FrequencySeries and TimeSeries

    ht_a, ht_b = pycbc.types.TimeSeries(ht_a, delta_t=1/sampling_frequency), pycbc.types.TimeSeries(ht_b, delta_t=1/sampling_frequency)
    ht =  pycbc.types.TimeSeries(ht, delta_t=1/sampling_frequency)
    h = ht.to_frequencyseries(delta_f=ht.delta_f)  #pycbc.types.FrequencySeries(ht, delta_f=delta_f)

    # Plotting the waveform injected in the detectors

    fig, ax = plt.subplots(1,3, figsize=(18, 6), sharex=True)
    fig.suptitle('\\textbf{WAVEFORM INJECTIONS (%s)}'%det_list[i])
    ax[0].plot(ht_a.sample_times, ht_a, label='$A$')
    ax[0].set_xlabel('Time $t$')
    ax[0].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
    ax[0].set_title('SINGLES A')
    ax[1].plot(ht_b.sample_times, ht_b, label='$B$')
    ax[1].set_xlabel('Time $t$')
    ax[1].set_ylabel('Strain $h=F_{+}h_{+}+F_{\\times}h_{\\times}$')
    ax[1].set_title('SINGLES B')
    ax[2].plot(ht.sample_times, ht)
    ax[2].set_xlabel('Time $t$')
    ax[2].set_ylabel('Strain $h=h_{\mathrm{A}}+h_{\mathrm{B}}$')
    ax[2].set_title('PAIRS with $\\Delta t_C=$%ss (%s)'%(np.round(delta_b,5), det_list[i]))
    plt.savefig('git_overlap/src/output/overlap_injection/WAVEFORM INJECTIONS (%s).png'%det_list[i])
    plt.close()