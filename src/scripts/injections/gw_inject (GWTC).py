# Generate BBH Injections (SINGLES/ PAIRS) resembling merger events and inject into detectors (H1, L1, V1) 

import pycbc
import numpy as np
import pycbc.types
import pycbc.waveform
import pycbc.detector
import matplotlib.pyplot as plt
from pesummary.io import read
from pycbc.frame import write_frame
from pycbc.waveform.utils import taper_timeseries

# Preamble

delta_f = 1
duration = 8
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

# Reading the posterior samples from GWTC release, and indexing the MAP values (https://zenodo.org/record/6513631)

injection_a = read('git_overlap/src/data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']   # Loading the GW150914 Posterior distributions
injection_b = read('git_overlap/src/data/IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']   # Loading the GW170814 Posterior distributions

nmap_a = np.argmax(injection_a['log_likelihood']+injection_a['log_prior'])   # Maximum A Posteriori values
nmap_b = np.argmax(injection_b['log_likelihood']+injection_b['log_prior'])   # Maximum A Posteriori values

def inject_singles(injection, nmap, start_time):
    """Generate PyCBC time domain SINGLES waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""

    waveform_params = {
        'approximant': 'IMRPhenomPv2',
        'mass1': injection['mass_1'][nmap],
        'mass2': injection['mass_2'][nmap],
        'spin1x': injection['spin_1x'][nmap],
        'spin1y': injection['spin_1y'][nmap],
        'spin1z': injection['spin_1z'][nmap],
        'spin2x': injection['spin_2x'][nmap],
        'spin2y': injection['spin_2y'][nmap],
        'spin2z': injection['spin_2z'][nmap],
        'distance': injection['luminosity_distance'][nmap],
        'inclination': injection['iota'][nmap],
        'coa_phase': injection['phase'][nmap],
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency
    }

    hp, hc = pycbc.waveform.get_td_waveform(**waveform_params)
    
    det = {}
    ifo_signal = {}
    for ifo in ['H1', 'L1', 'V1']:
        det[ifo] = pycbc.detector.Detector(ifo)
        ifo_signal[ifo] = det[ifo].project_wave(hp, hc, injection['ra'][nmap], injection['dec'][nmap], injection['psi'][nmap])
        wf = pycbc.types.TimeSeries(ifo_signal[ifo].data, delta_t=ifo_signal[ifo].delta_t, epoch=ifo_signal[ifo].start_time)

        diff_start = wf.sample_times[0] - int(wf.sample_times[0])
        dlen_start = round(wf.sample_rate * (1 + diff_start))
        wf_strain_start = np.concatenate((np.zeros(dlen_start), wf.data))
        wf_stime_start = np.concatenate((np.arange(wf.sample_times[0] - wf.delta_t, wf.sample_times[0] - (dlen_start + 1) * wf.delta_t, -wf.delta_t)[::-1], wf.sample_times))
        wf_start = pycbc.types.TimeSeries(wf_strain_start, delta_t=wf.delta_t, epoch=wf_stime_start[0])
        wf_start.start_time = wf.start_time

        diff_end = np.ceil(wf.sample_times[-1]) - (wf.sample_times[-1] + wf.delta_t)
        nlen_end = round(len(wf) + wf.sample_rate * (2 + diff_end))
        wf_end = pycbc.types.TimeSeries(wf.data, delta_t=wf.delta_t, epoch=wf.start_time)
        wf_end.resize(nlen_end)

        dur = wf_end.duration
        wf_end.resize(int(round(wf_end.sample_rate * np.power(2, np.ceil(np.log2(dur))))))

        ifo_signal[ifo] = wf_end

    ht_H1, ht_L1, ht_V1 = ifo_signal['H1'], ifo_signal['L1'], ifo_signal['V1']

    ht_H1.start_time = ht_L1.start_time = ht_V1.start_time = start_time

    ht_H1 = taper_timeseries(ht_H1, tapermethod='TAPER_STARTEND', return_lal=False)
    ht_L1 = taper_timeseries(ht_L1, tapermethod='TAPER_STARTEND', return_lal=False)
    ht_V1 = taper_timeseries(ht_V1, tapermethod='TAPER_STARTEND', return_lal=False)

    ht_H1 = pycbc.types.TimeSeries(ht_H1, delta_t=1 / ht_H1.sample_rate, epoch=round(ht_H1.sample_times[0]))
    ht_L1 = pycbc.types.TimeSeries(ht_L1, delta_t=1 / ht_L1.sample_rate, epoch=round(ht_L1.sample_times[0]))
    ht_V1 = pycbc.types.TimeSeries(ht_V1, delta_t=1 / ht_V1.sample_rate, epoch=round(ht_V1.sample_times[0]))

    return ht_H1, ht_L1, ht_V1

def inject_pairs(injection_a, nmap_a, start_time_a, injection_b, nmap_b, start_time_b):
    """Generate PyCBC time domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""

    waveform_params_a = {
        'approximant': 'IMRPhenomPv2',
        'mass1': injection_a['mass_1'][nmap_a],
        'mass2': injection_a['mass_2'][nmap_a],
        'spin1x': injection_a['spin_1x'][nmap_a],
        'spin1y': injection_a['spin_1y'][nmap_a],
        'spin1z': injection_a['spin_1z'][nmap_a],
        'spin2x': injection_a['spin_2x'][nmap_a],
        'spin2y': injection_a['spin_2y'][nmap_a],
        'spin2z': injection_a['spin_2z'][nmap_a],
        'distance': injection_a['luminosity_distance'][nmap_a],
        'inclination': injection_a['iota'][nmap_a],
        'coa_phase': injection_a['phase'][nmap_a],
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency
    }

    waveform_params_b = {
        'approximant': 'IMRPhenomPv2',
        'mass1': injection_b['mass_1'][nmap_b],
        'mass2': injection_b['mass_2'][nmap_b],
        'spin1x': injection_b['spin_1x'][nmap_b],
        'spin1y': injection_b['spin_1y'][nmap_b],
        'spin1z': injection_b['spin_1z'][nmap_b],
        'spin2x': injection_b['spin_2x'][nmap_b],
        'spin2y': injection_b['spin_2y'][nmap_b],
        'spin2z': injection_b['spin_2z'][nmap_b],
        'distance': injection_b['luminosity_distance'][nmap_b],
        'inclination': injection_b['iota'][nmap_b],
        'coa_phase': injection_b['phase'][nmap_b],
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency
    }

    det_a = {ifo: pycbc.detector.Detector(ifo) for ifo in ['H1', 'L1', 'V1']}
    det_b = {ifo: pycbc.detector.Detector(ifo) for ifo in ['H1', 'L1', 'V1']}
    ifo_signal_a, ifo_signal_b = {}, {}

    for ifo in ['H1', 'L1', 'V1']:
        hp_a, hc_a = pycbc.waveform.get_td_waveform(**waveform_params_a)
        ifo_signal_a[ifo] = det_a[ifo].project_wave(hp_a, hc_a, injection_a['ra'][nmap_a], injection_a['dec'][nmap_a], injection_a['psi'][nmap_a])
        wf_a = pycbc.types.TimeSeries(ifo_signal_a[ifo].data, delta_t=ifo_signal_a[ifo].delta_t, epoch=ifo_signal_a[ifo].start_time)
        wf_a = preprocess_waveform(wf_a)

        hp_b, hc_b = pycbc.waveform.get_td_waveform(**waveform_params_b)
        ifo_signal_b[ifo] = det_b[ifo].project_wave(hp_b, hc_b, injection_b['ra'][nmap_b], injection_b['dec'][nmap_b], injection_b['psi'][nmap_b])
        wf_b = pycbc.types.TimeSeries(ifo_signal_b[ifo].data, delta_t=ifo_signal_b[ifo].delta_t, epoch=ifo_signal_b[ifo].start_time)
        wf_b = preprocess_waveform(wf_b)

        ifo_signal_a[ifo] = wf_a
        ifo_signal_b[ifo] = wf_b

    ht_H1_a, ht_L1_a, ht_V1_a = ifo_signal_a['H1'], ifo_signal_a['L1'], ifo_signal_a['V1']
    ht_H1_b, ht_L1_b, ht_V1_b = ifo_signal_b['H1'], ifo_signal_b['L1'], ifo_signal_b['V1']

    ht_H1_a, ht_L1_a, ht_V1_a = preprocess_timeseries(ht_H1_a), preprocess_timeseries(ht_L1_a), preprocess_timeseries(ht_V1_a)
    ht_H1_b, ht_L1_b, ht_V1_b = preprocess_timeseries(ht_H1_b), preprocess_timeseries(ht_L1_b), preprocess_timeseries(ht_V1_b)

    hf_a_H1, hf_a_L1, hf_a_V1 = ht_H1_a.to_frequencyseries(), ht_L1_a.to_frequencyseries(), ht_V1_a.to_frequencyseries()
    hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_b.to_frequencyseries(), ht_L1_b.to_frequencyseries(), ht_V1_b.to_frequencyseries()

    hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (start_time_b - start_time_a) * hf_b_L1.sample_frequencies)
    hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (start_time_b - start_time_a) * hf_b_L1.sample_frequencies)
    hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (start_time_b - start_time_a) * hf_b_V1.sample_frequencies)

    ht_H1, ht_L1, ht_V1 = hf_H1.to_timeseries(), hf_L1.to_timeseries(), hf_V1.to_timeseries()
    ht_H1, ht_L1, ht_V1 = preprocess_waveform(ht_H1), preprocess_waveform(ht_L1), preprocess_waveform(ht_V1)
    ht_H1.start_time, ht_L1.start_time, ht_V1.start_time = start_time_a, start_time_a, start_time_a
    ht_H1, ht_L1, ht_V1 = preprocess_timeseries(ht_H1), preprocess_timeseries(ht_L1), preprocess_timeseries(ht_V1)
    
    return ht_H1, ht_L1, ht_V1

def preprocess_waveform(waveform):

    diff_start = waveform.sample_times[0] - int(waveform.sample_times[0])
    dlen_start = round(waveform.sample_rate * (1 + diff_start))
    wf_strain_start = np.concatenate((np.zeros(dlen_start), waveform.data))
    wf_stime_start = np.concatenate((np.arange(waveform.sample_times[0] - waveform.delta_t, waveform.sample_times[0] - (dlen_start + 1) * waveform.delta_t, -waveform.delta_t)[::-1], waveform.sample_times))
    wf_start = pycbc.types.TimeSeries(wf_strain_start, delta_t=waveform.delta_t, epoch=wf_stime_start[0])
    wf_start.start_time = waveform.start_time

    diff_end = np.ceil(waveform.sample_times[-1]) - (waveform.sample_times[-1] + waveform.delta_t)
    nlen_end = round(len(waveform) + waveform.sample_rate * (2 + diff_end))
    wf_end = pycbc.types.TimeSeries(waveform.data, delta_t=waveform.delta_t, epoch=waveform.start_time)
    wf_end.resize(nlen_end)

    dur = wf_end.duration
    wf_end.resize(int(round(wf_end.sample_rate * np.power(2, np.ceil(np.log2(dur))))))

    waveform_processed = pycbc.types.TimeSeries(wf_end, delta_t=1 / wf_end.sample_rate, epoch=round(wf_end.sample_times[0]))
    waveform_processed.start_time = waveform.start_time

    return waveform_processed

def preprocess_timeseries(timeseries):

    timeseries = taper_timeseries(timeseries, tapermethod='TAPER_STARTEND', return_lal=False)
    timeseries = pycbc.types.TimeSeries(timeseries, delta_t=1 / timeseries.sample_rate, epoch=round(timeseries.sample_times[0]))
    
    return timeseries

start_time_a = injection_a['geocent_time'][nmap_a]-duration+2
start_time_b = injection_a['geocent_time'][nmap_a]-duration+2+np.random.uniform(-2,2)

# Generating SINGLES A & B

ht_a_H1, ht_a_L1, ht_a_V1 = inject_singles(injection_a, nmap_a, start_time_a)

write_frame("git_overlap/src/output/injections/SINGLES_A_H1.gwf", "H1:PyCBC_Injection", ht_a_H1)
write_frame("git_overlap/src/output/injections/SINGLES_A_L1.gwf", "L1:PyCBC_Injection", ht_a_L1)
write_frame("git_overlap/src/output/injections/SINGLES_A_V1.gwf", "V1:PyCBC_Injection", ht_a_V1)

ht_b_H1, ht_b_L1, ht_b_V1 = inject_singles(injection_b, nmap_b, start_time_b)

write_frame("git_overlap/src/output/injections/SINGLES_B_H1.gwf", "H1:PyCBC_Injection", ht_b_H1)
write_frame("git_overlap/src/output/injections/SINGLES_B_L1.gwf", "L1:PyCBC_Injection", ht_b_L1)
write_frame("git_overlap/src/output/injections/SINGLES_B_V1.gwf", "V1:PyCBC_Injection", ht_b_V1)

# Generating PAIRS

ht_H1, ht_L1, ht_V1 = inject_pairs(injection_a, nmap_a, start_time_a, injection_b, nmap_b, start_time_b)

write_frame("git_overlap/src/output/injections/PAIRS_H1.gwf", "H1:PyCBC_Injection", ht_H1)
write_frame("git_overlap/src/output/injections/PAIRS_L1.gwf", "L1:PyCBC_Injection", ht_L1)
write_frame("git_overlap/src/output/injections/PAIRS_V1.gwf", "V1:PyCBC_Injection", ht_V1)
