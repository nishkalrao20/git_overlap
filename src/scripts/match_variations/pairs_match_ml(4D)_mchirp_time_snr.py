#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# 4 Dimensional (Chirp mass, Symmetric mass ratio, Redshifted Lens Mass, Impacgt Parameter) Nelder Mead maximization of Match on PAIRS waveforms, by varying SNR of SINGLES_B

import pycbc
import bilby
import pickle
import pycbc.psd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append("/home/nishkal.rao/GWMAT/pnt_Ff_lookup_table/src/cythonized_pnt_lens_class")   
import cythonized_pnt_lens_class as pnt_lens_cy 

# Setting some constants

delta_f = 1
duration = 100
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048
N_iter = int(12)   # Number of iterations of the initial points

# Matplotlib rcParams

plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

def mconv(mchirp, eta):
    """Calculates the masses given the chirp mass and symmetric mass ratio."""
    
    mtotal = mchirp * np.power(eta, -3/5)
    mass_1 = mtotal*(1+np.sqrt(1-4*eta))/2
    mass_2 = mtotal*(1-np.sqrt(1-4*eta))/2 

    return mass_1, mass_2

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

with open('/home/nishkal.rao/git_overlap/src/data/point_lens_Ff_lookup_table_Geo_relErr_1p0.pkl', 'rb') as f:
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

        return ht_lens

def inject_pairs(injection_parameters_a, injection_parameters_b):
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

    H1_strain, L1_strain, V1_strain = ifos[0].time_domain_strain, ifos[1].time_domain_strain, ifos[2].time_domain_strain

    # Generating PyCBC TimeSeries from the strain array, setting the start times to the geocenter time, and creating the dictionary of waveforms 

    ht_H1, ht_L1, ht_V1 = pycbc.types.TimeSeries(H1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain, delta_t = 1/sampling_frequency)
    ht_H1.start_time, ht_L1.start_time, ht_V1.start_time = injection_parameters_a['geocent_time']-duration+2, injection_parameters_a['geocent_time']-duration+2, injection_parameters_a['geocent_time']-duration+2
    ht = {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

    return ht

def read_psd(det):
    """Reading the PSD files for the detectors"""

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('/home/nishkal.rao/git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('/home/nishkal.rao/git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('/home/nishkal.rao/git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

def mismatch(h, mchirp, eta, Ml_z, y, injection_parameters, det, psd):
    """Calculating the match/overlap defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800 or y<0.01 or y>5.00 or Ml_z<10 or Ml_z>1e5:
        log_mismatch = 1e6
    
    else:
        # Set the injection parameters and generate the template waveforms

        injection_parameters['mass_1'], injection_parameters['mass_2'] = mass_1, mass_2
        injection_parameters['mass_ratio'] = mass_2/mass_1

        h_templ = inject_wf_lens(injection_parameters, Ml_z, y)[det] 
        
        # Resize the templates to the length of the waveform and interpolate the PSD
        
        h.resize(max(len(h), len(h_templ)))
        h_templ.resize(max(len(h), len(h_templ)))
        psd = pycbc.psd.interpolate(psd, 1/h.duration)

        # Evaluating the matchedfilter match through pycbc

        match = pycbc.filter.matchedfilter.match(h, h_templ, low_frequency_cutoff=minimum_frequency, psd=psd, 
                                                 high_frequency_cutoff=maximum_frequency, v1_norm=None, v2_norm=None)[0]
        log_mismatch = np.log10(1-match)   # Defining the log(mismatch)
    
    return log_mismatch 

def match(injection_parameters_a, injection_parameters_b, det, N_iter, k):
    """Initializing the Mismatch function over an iniitial array of uniformily distributed variables around the injected parameters"""

    mchirp_a, q_a = injection_parameters_a['chirp_mass'], injection_parameters_a['mass_ratio']
    mchirp_b, q_b = injection_parameters_b['chirp_mass'], injection_parameters_b['mass_ratio']

    mass_1_a, mass_2_a = mchirp_a*np.power(1+q_a,1/5)/np.power(q_a,3/5), mchirp_a*np.power(1+q_a,1/5)*np.power(q_a,2/5)
    mass_1_b, mass_2_b = mchirp_b*np.power(1+q_b,1/5)/np.power(q_b,3/5), mchirp_b*np.power(1+q_b,1/5)*np.power(q_b,2/5)
    eta_a, eta_b = (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)

    mchirp = (mchirp_a*np.abs(waveform_metadata_a[det]['optimal_SNR'])+mchirp_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    eta = (eta_a*np.abs(waveform_metadata_a[det]['optimal_SNR'])+eta_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
        
    # Generating the PAIRS waveforms and Interpolating the PSD
    
    h, psd = inject_pairs(injection_parameters_a, injection_parameters_b)[det], read_psd(det)

    # Distributing the initial array of points as an uniform continuous rv around the weighted average of the injected values

    mchirp_0 = sp.stats.uniform.rvs(size = int(1e4))*2*mchirp
    eta_0 = sp.stats.uniform.rvs(size = int(1e4))*2*eta
    Ml_z_0 = sp.stats.loguniform.rvs(10, 1e4, size=int(1e4))
    y_0 = sp.stats.uniform.rvs(0.01, 2, size = int(1e4))
    
    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (0.01<y_0) & (y_0<5.00)

    # Appending an array of initial points spread as a uniform rv around the weighted average of the injected values

    mchirp_arr = np.append(mchirp, np.random.choice(mchirp_0[idx], int(N_iter)-1))
    eta_arr = np.append(eta, np.random.choice(eta_0[idx], int(N_iter)-1))
    Ml_z_arr = np.append(np.mean(Ml_z_0), np.random.choice(Ml_z_0[idx], int(N_iter)-1))
    y_arr = np.append(np.mean(y_0), np.random.choice(y_0[idx], int(N_iter)-1))

    # Initializing a function (returning log(mismatch)) for minimizing, with parameters weighted over the SNRs

    injection_parameters = injection_parameters_a.copy()
    for key in injection_parameters:
        injection_parameters[key] = (injection_parameters_a[key]*np.abs(waveform_metadata_a[det]['optimal_SNR'])+injection_parameters_b[key]*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    fun_mismatch = lambda x: mismatch(h, x[0], x[1], x[2], x[3], injection_parameters, det, psd)
    
    # Minimizing the mismatch over all the initial set of points    

    log_mismatch, params = [], []
    for i in range(int(N_iter)):
        res = sp.optimize.minimize(fun_mismatch, (mchirp_arr[i], eta_arr[i], Ml_z_arr[i], y_arr[i]), method='Nelder-Mead', options={'adaptive':True, 'return_all': True}, tol=1e-3) 
        log_mismatch.append(res.fun)
        params.append(res.allvecs)
    match = 1 - np.power(10, log_mismatch)

    return log_mismatch, match, params

# Importing the waveform metadata and setting the intitial parameters

waveform_metadata_a = pickle.load(open('/home/nishkal.rao/git_overlap/src/output/match_variations/match_mchirp_time_snr/injections/GW Waveform A Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
mchirp_a, q_a = waveform_metadata_a['H1']['parameters']['chirp_mass'], waveform_metadata_a['H1']['parameters']['mass_ratio']
mass_1_a, mass_2_a = mchirp_a*np.power(1+q_a,1/5)/np.power(q_a,3/5), mchirp_a*np.power(1+q_a,1/5)*np.power(q_a,2/5)
eta_a = (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2)
luminosity_distance_a = waveform_metadata_a['H1']['parameters']['luminosity_distance']
snr_a = np.abs(waveform_metadata_a['H1']['optimal_SNR'])

N_sampl = 125
waveform_metadata_b = []
mass_1_b, mass_2_b, mchirp_b, eta_b, q_b, luminosity_distance_b, snr_b, delta_tc = np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl)
for k in range(N_sampl):
    waveform_metadata_b.append(pickle.load(open('/home/nishkal.rao/git_overlap/src/output/match_variations/match_mchirp_time_snr/injections/GW Waveform B Meta Data %s.pkl'%(k+1), 'rb')))   # Importing Waveform Meta Data
    mchirp_b[k], q_b[k] = waveform_metadata_b[k]['H1']['parameters']['chirp_mass'], waveform_metadata_b[k]['H1']['parameters']['mass_ratio']
    mass_1_b[k], mass_2_b[k] = mchirp_b[k]*np.power(1+q_b[k],1/5)/np.power(q_b[k],3/5), mchirp_b[k]*np.power(1+q_b[k],1/5)*np.power(q_b[k],2/5)
    eta_b[k] = (mass_1_b[k]*mass_2_b[k])/np.power(mass_1_b[k]+mass_2_b[k], 2)
    luminosity_distance_b[k], snr_b[k] = waveform_metadata_b[k]['H1']['parameters']['luminosity_distance'], np.abs(waveform_metadata_b[k]['H1']['optimal_SNR'])
    delta_tc[k] = waveform_metadata_b[k]['H1']['parameters']['geocent_time']-waveform_metadata_a['H1']['parameters']['geocent_time']

# Evaluating the log(mismatch) values for the PAIRS

k = int(sys.argv[1])
log_mismatch_H1, match_H1, params_H1 = match(waveform_metadata_a['H1']['parameters'], waveform_metadata_b[k]['H1']['parameters'], 'H1', int(N_iter), k)

# Choosing the convergent values from the parameter space

mchirp_H1 = [[params_H1[i][j][0] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
eta_H1 = [[params_H1[i][j][1] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
Ml_z_H1 = [[params_H1[i][j][2] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
y_H1 = [[params_H1[i][j][3] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]

mchirp_arr_H1 = [mchirp_H1[i][len(mchirp_H1[i])-1] for i in range(N_iter)]
eta_arr_H1 = [eta_H1[i][len(eta_H1[i])-1] for i in range(N_iter)]
Ml_z_arr_H1 = [Ml_z_H1[i][len(Ml_z_H1[i])-1] for i in range(N_iter)]
y_arr_H1 = [y_H1[i][len(y_H1[i])-1] for i in range(N_iter)]

min_idx, max_idx = np.argmin(log_mismatch_H1), np.argmax(match_H1)
log_mismatch_f_H1, match_f_H1 = log_mismatch_H1[min_idx], match_H1[max_idx]
match_H1[np.isneginf(match_H1)]=0

mchirpf_H1, etaf_H1, Ml_zf_H1, yf_H1 = mchirp_arr_H1[min_idx], eta_arr_H1[min_idx], Ml_z_arr_H1[min_idx], y_arr_H1[min_idx]
mass_1f_H1, mass_2f_H1 = mconv(mchirpf_H1, etaf_H1)
np.savetxt('/home/nishkal.rao/git_overlap/src/output/match_variations/match_ml_mchirp_time_snr/outputs/PAIRS(4D) %s.csv'%(k+1), np.column_stack((match_f_H1, mchirpf_H1, etaf_H1, Ml_zf_H1, yf_H1)), delimiter=',')