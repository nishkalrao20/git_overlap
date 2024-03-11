import pycbc
import bilby
import pickle
import numpy as np 
import scipy as sp 

import sys
sys.path.append("GWMAT/pnt_Ff_lookup_table/src/cythonized_pnt_lens_class")   
import cythonized_pnt_lens_class as pnt_lens_cy 

import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 16,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

# Setting some constants

delta_f = 1
duration = 100
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

def inject_wf(injection_parameters):
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

    # Generating PyCBC TimeSeries from the strain array, setting the start times to the geocenter time, and creating the dictionary of waveforms 

    ht_H1, ht_L1, ht_V1 = pycbc.types.TimeSeries(H1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain, delta_t = 1/sampling_frequency)
    ht_H1.start_time, ht_L1.start_time, ht_V1.start_time = injection_parameters['geocent_time']-duration+2, injection_parameters['geocent_time']-duration+2, injection_parameters['geocent_time']-duration+2
    ht = {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

    return ht

with open('git_overlap/src/data/point_lens_Ff_lookup_table_Geo_relErr_1p0.pkl', 'rb') as f:
    Ff_grid = pickle.load(f)
    ys_grid = np.array([Ff_grid[str(i)]['y'] for i in range(len(Ff_grid))])
    ws_grid = Ff_grid['0']['ws']

def pnt_Ff_lookup_table(ys_grid, ws_grid, fs, Mlz, yl, extrapolate=True):
    wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
    wc = pnt_lens_cy.wc_geo_re1p0(yl)

    wfs_1 = wfs[wfs <= np.min(ws_grid)]
    Ffs_1 = np.array([1]*len(wfs_1))

    wfs_2 = wfs[(wfs > np.min(ws_grid))&(wfs <= np.max(ws_grid))]
    wfs_2_wave = wfs_2[wfs_2 <= wc]
    wfs_2_geo = wfs_2[wfs_2 > wc]

    i_y  = np.argmin(np.abs(ys_grid - yl))
    ws = Ff_grid[str(i_y)]['ws']
    Ffs = Ff_grid[str(i_y)]['Ffs_real'] + 1j*Ff_grid[str(i_y)]['Ffs_imag']
    fill_val = ['interpolate', 'extrapolate'][extrapolate]
    i_Ff = sp.interpolate.interp1d(ws, Ffs, fill_value=fill_val)
    Ffs_2_wave = i_Ff(wfs_2_wave)

    Ffs_2_geo = np.array([pnt_lens_cy.point_Fw_geo(w, yl) for w in wfs_2_geo])

    wfs_3 = wfs[wfs > np.max(ws_grid) ]
    Ffs_3 = np.array([pnt_lens_cy.point_Fw_geo(w, Mlz) for w in wfs_3])

    Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))

    return Ffs, wfs

def inject_wf_lens(injection_parameters, Ml_z, y):
    """Generate microlensed time domain waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors (using GWMAT)"""

    ht = inject_wf(injection_parameters)

    if round(Ml_z) == 0:
        return ht
    else:
        hf_H1, hf_L1, hf_V1 = ht['H1'].to_frequencyseries(), ht['L1'].to_frequencyseries(), ht['V1'].to_frequencyseries()

        Ff, wfs = pnt_Ff_lookup_table(ys_grid=ys_grid, ws_grid=ws_grid, fs=hf_H1.sample_frequencies, Mlz=Ml_z, yl=y)

        hfl_H1, hfl_L1, hfl_V1 = pycbc.types.FrequencySeries(Ff*hf_H1, delta_f = hf_H1.delta_f), pycbc.types.FrequencySeries(Ff*hf_L1, delta_f = hf_L1.delta_f), pycbc.types.FrequencySeries(Ff*hf_V1, delta_f = hf_V1.delta_f)

        htl_H1, htl_L1, htl_V1 = hfl_H1.to_timeseries(), hfl_L1.to_timeseries(), hfl_V1.to_timeseries()
        htl_H1.start_time, htl_L1.start_time, htl_V1.start_time = injection_parameters['geocent_time']-duration+2, injection_parameters['geocent_time']-duration+2, injection_parameters['geocent_time']-duration+2
        ht_lens = {'H1': htl_H1, 'L1': htl_L1, 'V1': htl_V1}

        return ht_lens, Ff, wfs

waveform_metadata = pickle.load(open('git_overlap/src/output/injections/GW Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
injection_parameters = waveform_metadata['H1']['parameters']

ht = inject_wf(injection_parameters)
ht_lens, Ff, wfs = inject_wf_lens(injection_parameters, 1e4, 0.3)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], 'k--', linewidth=1, label='$\\rm{Unlensed}$')
ax.plot(ht_lens['H1'].sample_times, ht_lens['H1'], 'm-', linewidth=1, label='$\\rm{Microlensed}$')
ax.set_xlabel('Time $[s]$')
ax.set_ylabel('Strain $h$')
ax.set_xlim(injection_parameters['geocent_time']-0.7, injection_parameters['geocent_time']+0.1)
ax.legend()
ax.grid(True)
plt.savefig('git_overlap/src/output/match_ml/plots/MicrolensedSignals1.pdf', format='pdf', bbox_inches="tight")
plt.show()
plt.close()

Mlz = int(1e4)
fig, ax = plt.subplots(figsize=(16, 9))
y = [0.01, 0.1, 0.3, 1, 2]
colors=['r','g','b','y','k']
y_labels = ['0.01', '0.10', '0.30', '1.00', '2.00']
for idx, y in enumerate(y):
    ht_lens, Ff, wfs = inject_wf_lens(injection_parameters, 1e4, y)
    Ff = pycbc.types.FrequencySeries(Ff, delta_f = ht_lens['H1'].to_frequencyseries().delta_f)
    ax.plot(Ff.sample_frequencies, np.abs(Ff), 'g-', linewidth=1, label=y_labels[idx], color=colors[idx])
ax.set_xscale('log')
ax.set_xlabel('Frequency $f$ [Hz]')
ax.set_ylabel('Amplification Factor $\\big|F(f)\\big|$')
ax.grid(True)
ax.legend()
plt.savefig('git_overlap/src/output/match_ml/plots/MicrolensedSignals2.pdf', format='pdf', bbox_inches="tight")
plt.show()
plt.close()