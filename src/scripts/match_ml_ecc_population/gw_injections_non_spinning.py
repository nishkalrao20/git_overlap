#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# Generate Overlapping BBH Injections from a population and inject into detectors (H1, L1, V1)

import sys
import bilby
import pycbc
import pickle
import deepdish
import pycbc.psd
import pycbc.types
import gwpopulation
import pycbc.waveform
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setting some constants

duration = 100
minimum_frequency = 20
reference_frequency = 50  # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048
delta_f = 1/sampling_frequency

def snr(injection_parameters):
    """Generate time domain waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""

    waveform_generator = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, start_time = injection_parameters['geocent_time']-duration+2,
                                                    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                    waveform_arguments = {'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])   # Initializing the Detectors

    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = minimum_frequency, sampling_frequency/2
    ifos.set_strain_data_from_zero_noise(sampling_frequency = sampling_frequency, duration = duration, start_time = injection_parameters['geocent_time']-duration+2)

    # Injecting the SINGLES GW signal into H1, L1, and V1 using bilby and extrapolating the strain data in the time domain

    ifos.inject_signal(waveform_generator = waveform_generator, parameters = injection_parameters)  
    H1_strain, L1_strain, V1_strain = ifos[0].time_domain_strain, ifos[1].time_domain_strain, ifos[2].time_domain_strain

    snrs = [ifo.meta_data["optimal_SNR"] for ifo in ifos]
    network_snr = np.sqrt(np.sum([i ** 2 for i in snrs]))

    return network_snr

def snr_pairs(injection_parameters_a, injection_parameters_b):
    """Generate time domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""

    waveform_generator_a = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, start_time = injection_parameters_a['geocent_time']-duration+2,
                                                     frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                     waveform_arguments = {'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

    waveform_generator_b = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, start_time = injection_parameters_a['geocent_time']-duration+2,
                                                     frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, 
                                                     waveform_arguments = {'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])   # Initializing the Detectors

    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = minimum_frequency, sampling_frequency/2
    ifos.set_strain_data_from_zero_noise(sampling_frequency = sampling_frequency, duration = duration, start_time = injection_parameters_a['geocent_time']-duration+2)

    # Injecting the SINGLES GW signal into H1, L1, and V1 using bilby and extrapolating the strain data in the time domain

    ifos.inject_signal(waveform_generator = waveform_generator_a, parameters = injection_parameters_a)    # PAIRS (A)
    ifos.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b)    # PAIRS (B)

    snrs = [ifo.meta_data["matched_filter_SNR"] for ifo in ifos]
    network_snr = np.sqrt(np.sum([i ** 2 for i in snrs]))

    return network_snr

'''
def read_psd(det):
    """Reading the PSD files for the detectors"""

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/psd_aLIGO_O4high.txt', int(duration*sampling_frequency/2+1), delta_f, minimum_frequency, is_asd_file=False)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/psd_aLIGO_O4high.txt', int(duration*sampling_frequency/2+1), delta_f, minimum_frequency, is_asd_file=False)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/psd_aVirgo_O4high_NEW.txt', int(duration*sampling_frequency/2+1), delta_f, minimum_frequency, is_asd_file=False)

    return psd

# Selecting SNRs > 8

def snr(injection_parameters):
    """Evaluates the matched filter signal-to-noise ratio (through PyCBC) for the set of waveforms generated by the injection parameters"""

    ht = inject_wf(injection_parameters)

    hf, snrs = {}, {}
    for det in ht.keys():
        hf[det] = ht[det].to_frequencyseries()
        
        hf_templ, _ = pycbc.waveform.get_fd_waveform(approximant = 'IMRPhenomPv2', mass1 = injection_parameters['mass_1'], mass2 = injection_parameters['mass_2'], distance = injection_parameters['luminosity_distance'], inclination = injection_parameters['incl'], coa_phase = injection_parameters['phase'], delta_f = delta_f, f_lower = minimum_frequency)
        
        print(len(hf[det]), len(hf_templ))
        
        hf[det].resize(max(len(hf[det]), len(hf_templ)))
        hf_templ.resize(max(len(hf[det]), len(hf_templ)))

        psd = read_psd(det)

        print(len(ht[det]), len(hf[det]), len(hf_templ), len(psd))

        snrs[det] = pycbc.filter.matched_filter(hf[det], hf_templ, psd=psd, low_frequency_cutoff=minimum_frequency)
        snrs[det] = max(abs(snrs[det]))

    return np.sqrt(snrs['H1']**2+snrs['L1']**2+snrs['V1']**2)

def snr_pairs(injection_parameters_a, injection_parameters_b):
    """Evaluates the matched filter signal-to-noise ratio (through PyCBC) for the set of waveforms generated by the injection parameters"""

    ht = inject_pairs(injection_parameters_a, injection_parameters_b)
    snr_a, snr_b = snr(injection_parameters_a), snr(injection_parameters_b)
    hf, snrs = {}, {}
    for det in ht.keys():
        hf[det] = ht[det].to_frequencyseries()
        if snr_a >= snr_b:
            hf_templ, _ = pycbc.waveform.get_fd_waveform(approximant = 'IMRPhenomPv2', mass1 = injection_parameters_a['mass_1'], mass2 = injection_parameters_a['mass_2'], distance = injection_parameters_a['luminosity_distance'], inclination = injection_parameters_a['incl'], coa_phase = injection_parameters_a['phase'], delta_f = delta_f, f_lower = minimum_frequency)
        else:
            hf_templ, _ = pycbc.waveform.get_fd_waveform(approximant = 'IMRPhenomPv2', mass1 = injection_parameters_b['mass_1'], mass2 = injection_parameters_b['mass_2'], distance = injection_parameters_b['luminosity_distance'], inclination = injection_parameters_b['incl'], coa_phase = injection_parameters_b['phase'], delta_f = delta_f, f_lower = minimum_frequency)
                
        hf[det].resize(max(len(hf[det]), len(hf_templ)))
        hf_templ.resize(max(len(hf[det]), len(hf_templ)))

        psd = read_psd(det)

        snrs[det] = pycbc.filter.matched_filter(hf[det], hf_templ, psd=psd, low_frequency_cutoff=minimum_frequency)
        snrs[det] = max(abs(snrs[det]))
        
    return np.sqrt(snrs['H1']**2+snrs['L1']**2+snrs['V1']**2)
'''

injection = dict(deepdish.io.load('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/injections/injections.hdf5')['injections'])

for key, val in injection.items():
    exec(key + '=val')
mchirp = (mass_1 * mass_2) ** (3 / 5) / (mass_1 + mass_2) ** (1 / 5)

num_points = int(len(mchirp) / 2)
idxs = np.random.choice(len(mchirp), size=(num_points, 2), replace=False)

unique_idxs, counts = np.unique(idxs[:, 0], return_counts=True)
mask = counts == 1
filtered_idxs = idxs[mask]

k = int(sys.argv[1])
queue = 100
filtered_idxs_mod = filtered_idxs[k*int(len(filtered_idxs)/queue):(k+1)*int(len(filtered_idxs)/queue)]

delta_tc_0 = np.logspace(0, 1, int(len(filtered_idxs)/2))/10
delta_tc = np.concatenate((-delta_tc_0, delta_tc_0))

for i, idx in enumerate(tqdm(filtered_idxs_mod)):

    mchirp_ratio = mchirp[idx[1]] / mchirp[idx[0]]

    if 0.5 <= mchirp_ratio <= 2:

        injection_parameters_a, injection_parameters_b = {}, {}
        
        for key, val in injection.items():
            injection_parameters_a[key], injection_parameters_b[key] = val[idx[0]], val[idx[1]]

        injection_parameters_a['a_1'], injection_parameters_a['a_2'] = 0, 0   #Non-Spinning
        injection_parameters_b['a_1'], injection_parameters_b['a_2'] = 0, 0   #Non-Spinning

        injection_parameters_b['geocent_time'] += delta_tc[np.random.randint(0, len(delta_tc))]
        start_time_a = injection_parameters_a['geocent_time'] - duration + 2
        start_time_b = injection_parameters_b['geocent_time'] - duration + 2
                
        injection_parameters_a['snr_det'] = np.abs(snr(injection_parameters_a))
        injection_parameters_b['snr_det'] = np.abs(snr(injection_parameters_b))

        snr_det = np.abs(snr_pairs(injection_parameters_a, injection_parameters_b))

        if 0.5 <= injection_parameters_b['snr_det']/injection_parameters_a['snr_det'] <= 2 and snr_det >= 8:

            waveform_generator_a = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency, start_time=start_time_a,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency,
                                    'minimum_frequency': minimum_frequency}
            )

            waveform_generator_b = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency, start_time=start_time_b,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency,
                                    'minimum_frequency': minimum_frequency}
            )

            ifos_a, ifos_b = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

            for det in [ifos_a, ifos_b]:
                for ifo in det:
                    ifo.minimum_frequency = minimum_frequency
                    ifo.maximum_frequency = sampling_frequency/2
                det.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=start_time_a)

            ifos_a.inject_signal(waveform_generator=waveform_generator_a, parameters=injection_parameters_a)
            with open('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/injections/GW Waveform A Meta Data %s.pkl'%(i+1), 'wb') as file:
                pickle.dump(ifos_a.meta_data, file)

            ifos_b.inject_signal(waveform_generator=waveform_generator_b, parameters=injection_parameters_b)
            with open('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/injections/GW Waveform B Meta Data %s.pkl'%(i+1), 'wb') as file:
                pickle.dump(ifos_b.meta_data, file)
