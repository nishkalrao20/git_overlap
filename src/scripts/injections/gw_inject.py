# Generate BBH Injections (SINGLES/ PAIRS) and inject into detectors (H1, L1, V1) 

import pycbc
import numpy as np
import pycbc.types
import pycbc.waveform
import pycbc.detector
import matplotlib.pyplot as plt
from pycbc.frame import write_frame
from gwpy.timeseries import TimeSeries
from pycbc.waveform.utils import taper_timeseries

# Preamble

delta_f = 1
duration = 8
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

def inject_singles(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, luminosity_distance, iota, phase, ra, dec, psi, start_time):
    """Generate PyCBC time domain SINGLES waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""

    waveform_params = {
        'approximant': 'IMRPhenomPv2',
        'mass1': mass1,
        'mass2': mass2,
        'spin1x': spin1x,
        'spin1y': spin1y,
        'spin1z': spin1z,
        'spin2x': spin2x,
        'spin2y': spin2y,
        'spin2z': spin2z,
        'distance': luminosity_distance,
        'inclination': iota,
        'coa_phase': phase,
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency
    }

    hp, hc = pycbc.waveform.get_td_waveform(**waveform_params)
    hp.start_time += start_time
    hc.start_time += start_time

    det = {}
    ifo_signal = {}
    for ifo in ['H1', 'L1', 'V1']:
        det[ifo] = pycbc.detector.Detector(ifo)
        ifo_signal[ifo] = det[ifo].project_wave(hp, hc, ra, dec, psi)
        wf = pycbc.types.TimeSeries(ifo_signal[ifo].data, delta_t=ifo_signal[ifo].delta_t, epoch=ifo_signal[ifo].start_time)
        ifo_signal[ifo] = preprocess_waveform(wf)

    ht_H1, ht_L1, ht_V1 = ifo_signal['H1'], ifo_signal['L1'], ifo_signal['V1']

    ht_H1.start_time = ht_L1.start_time = ht_V1.start_time = start_time

    ht_H1 = taper_timeseries(ht_H1, tapermethod='TAPER_STARTEND', return_lal=False)
    ht_L1 = taper_timeseries(ht_L1, tapermethod='TAPER_STARTEND', return_lal=False)
    ht_V1 = taper_timeseries(ht_V1, tapermethod='TAPER_STARTEND', return_lal=False)

    ht_H1 = pycbc.types.TimeSeries(ht_H1, delta_t=1 / ht_H1.sample_rate, epoch=round(ht_H1.sample_times[0]))
    ht_L1 = pycbc.types.TimeSeries(ht_L1, delta_t=1 / ht_L1.sample_rate, epoch=round(ht_L1.sample_times[0]))
    ht_V1 = pycbc.types.TimeSeries(ht_V1, delta_t=1 / ht_V1.sample_rate, epoch=round(ht_V1.sample_times[0]))

    return ht_H1, ht_L1, ht_V1

def inject_pairs(mass1_a, mass2_a, spin1x_a, spin1y_a, spin1z_a, spin2x_a, spin2y_a, spin2z_a, luminosity_distance_a, iota_a, phase_a, ra_a, dec_a, psi_a, start_time_a, mass1_b, mass2_b, spin1x_b, spin1y_b, spin1z_b, spin2x_b, spin2y_b, spin2z_b, luminosity_distance_b, iota_b, phase_b, ra_b, dec_b, psi_b, start_time_b):
    """Generate PyCBC time domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors"""

    waveform_params_a = {
        'approximant': 'IMRPhenomPv2',
        'mass1': mass1_a,
        'mass2': mass2_a,
        'spin1x': spin1x_a,
        'spin1y': spin1y_a,
        'spin1z': spin1z_a,
        'spin2x': spin2x_a,
        'spin2y': spin2y_a,
        'spin2z': spin2z_a,
        'distance': luminosity_distance_a,
        'inclination': iota_a,
        'coa_phase': phase_a,
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency
    }

    waveform_params_b = {
        'approximant': 'IMRPhenomPv2',
        'mass1': mass1_b,
        'mass2': mass2_b,
        'spin1x': spin1x_b,
        'spin1y': spin1y_b,
        'spin1z': spin1z_b,
        'spin2x': spin2x_b,
        'spin2y': spin2y_b,
        'spin2z': spin2z_b,
        'distance': luminosity_distance_b,
        'inclination': iota_b,
        'coa_phase': phase_b,
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency
    }

    det_a = {ifo: pycbc.detector.Detector(ifo) for ifo in ['H1', 'L1', 'V1']}
    det_b = {ifo: pycbc.detector.Detector(ifo) for ifo in ['H1', 'L1', 'V1']}
    ifo_signal_a, ifo_signal_b = {}, {}

    for ifo in ['H1', 'L1', 'V1']:
        hp_a, hc_a = pycbc.waveform.get_td_waveform(**waveform_params_a)
        hp_a.start_time += start_time_a
        hc_a.start_time += start_time_a

        ifo_signal_a[ifo] = det_a[ifo].project_wave(hp_a, hc_a, ra_a, dec_a, psi_a)
        wf_a = pycbc.types.TimeSeries(ifo_signal_a[ifo].data, delta_t=ifo_signal_a[ifo].delta_t, epoch=ifo_signal_a[ifo].start_time)
        wf_a = preprocess_waveform(wf_a)

        hp_b, hc_b = pycbc.waveform.get_td_waveform(**waveform_params_b)
        hp_b.start_time += start_time_b
        hc_b.start_time += start_time_b

        ifo_signal_b[ifo] = det_b[ifo].project_wave(hp_b, hc_b, ra_b, dec_b, psi_b)
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

    wf_end.resize(int(round(wf_end.sample_rate * np.power(2, np.ceil(np.log2(wf_end.duration))))))

    waveform_processed = pycbc.types.TimeSeries(wf_end, delta_t=1 / wf_end.sample_rate, epoch=round(wf_end.sample_times[0]))
    waveform_processed.start_time = waveform.start_time

    return waveform_processed 

def preprocess_timeseries(timeseries):

    timeseries = taper_timeseries(timeseries, tapermethod='TAPER_STARTEND', return_lal=False)
    timeseries = pycbc.types.TimeSeries(timeseries, delta_t=1 / timeseries.sample_rate, epoch=round(timeseries.sample_times[0]))
    
    return timeseries

# Generating SINGLES A & B

ht_a_H1, ht_a_L1, ht_a_V1 = inject_singles(mass1=30, mass2=40, spin1x=0, spin1y=0, spin1z=0.5, spin2x=0, spin2y=0, spin2z=0.5, luminosity_distance=500, iota=2.5, phase=0, ra=2.2, dec=-1.25, psi=1.75, start_time=1126259462.4116447)

ht_b_H1, ht_b_L1, ht_b_V1 = inject_singles(mass1=20, mass2=30, spin1x=0, spin1y=0, spin1z=0.75, spin2x=0, spin2y=0, spin2z=0.25, luminosity_distance=800, iota=1.5, phase=2, ra=1.2, dec=1.25, psi=2.75, start_time=1126259462.4116447+0.2)

# Generating PAIRS

ht_H1, ht_L1, ht_V1 = inject_pairs(mass1_a=30, mass2_a=40, spin1x_a=0, spin1y_a=0, spin1z_a=0.5, spin2x_a=0, spin2y_a=0, spin2z_a=0.5, luminosity_distance_a=500, iota_a=2.5, phase_a=0, ra_a=2.2, dec_a=-1.25, psi_a=1.75, start_time_a=1126259462.4116447, mass1_b=20, mass2_b=30, spin1x_b=0, spin1y_b=0, spin1z_b=0.75, spin2x_b=0, spin2y_b=0, spin2z_b=0.25, luminosity_distance_b=450, iota_b=1.5, phase_b=2, ra_b=1.2, dec_b=1.25, psi_b=2.75, start_time_b=1126259462.4116447+0.2)

# Saving the frame files

write_frame("git_overlap/src/output/injections/sarathi/SINGLES_A_H1.gwf", "H1:PyCBC_Injection", ht_a_H1)
write_frame("git_overlap/src/output/injections/sarathi/SINGLES_A_L1.gwf", "L1:PyCBC_Injection", ht_a_L1)
write_frame("git_overlap/src/output/injections/sarathi/SINGLES_A_V1.gwf", "V1:PyCBC_Injection", ht_a_V1)

write_frame("git_overlap/src/output/injections/sarathi/SINGLES_B_H1.gwf", "H1:PyCBC_Injection", ht_b_H1)
write_frame("git_overlap/src/output/injections/sarathi/SINGLES_B_L1.gwf", "L1:PyCBC_Injection", ht_b_L1)
write_frame("git_overlap/src/output/injections/sarathi/SINGLES_B_V1.gwf", "V1:PyCBC_Injection", ht_b_V1)

write_frame("git_overlap/src/output/injections/sarathi/PAIRS_H1.gwf", "H1:PyCBC_Injection", ht_H1)
write_frame("git_overlap/src/output/injections/sarathi/PAIRS_L1.gwf", "L1:PyCBC_Injection", ht_L1)
write_frame("git_overlap/src/output/injections/sarathi/PAIRS_V1.gwf", "V1:PyCBC_Injection", ht_V1)