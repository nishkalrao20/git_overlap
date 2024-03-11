# 3 Dimensional (Chirp mass, Symmetric mass ratio, Phase) Nelder Mead maximization of Match on PAIRS waveforms, by varying SNR of SINGLES_B

import os
import scipy
import pycbc
import bilby
import pickle
import pycbc.psd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

def mismatch(h, mchirp, eta, eff_spin, injection_parameters, det, psd):
    """Calculating the match/overlap defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or mass_1 < 2 or mass_2 < 2 or mass_1/mass_2 < 1/18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800 or eff_spin<0.01 or eff_spin>0.99:
        log_mismatch = 1e6
    
    else:
        # Set the injection parameters and generate the template waveforms

        injection_parameters['mass_1'], injection_parameters['mass_2'] = mass_1, mass_2
        injection_parameters['chirp_mass'] = (mass_1*mass_2)**(3/5)/(mass_1+mass_2)**(1/5)
        injection_parameters['mass_ratio'] = mass_2/mass_1
        injection_parameters['a_1'], injection_parameters['a_2'] = eff_spin, eff_spin

        h_templ = inject_wf(injection_parameters)[det] 
        
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
    """Initializing the Mismatch function over an iniitial array of truncated Gaussians around the injected parameters"""

    mass_1_a, mass_2_a = injection_parameters_a['mass_1'], injection_parameters_a['mass_2']
    mass_1_b, mass_2_b = injection_parameters_b['mass_1'], injection_parameters_b['mass_2']
    mchirp_a, mchirp_b = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5)
    eta_a, eta_b = (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)
    eff_spin_a, eff_spin_b = injection_parameters_a['a_1'], injection_parameters_b['a_1']

    mchirp = (mchirp_a*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+mchirp_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    eta = (eta_a*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+eta_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    eff_spin = (eff_spin_a*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+eff_spin_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))

    # Generating the PAIRS waveforms and Interpolating the PSD
    
    h, psd = inject_pairs(injection_parameters_a, injection_parameters_b)[det], read_psd(det)

    # Distributing the initial array of points as an uniform continuous rv around the weighted average of the injected values

    mchirp_0 = scipy.stats.uniform.rvs(size = int(1e4))*2*mchirp
    eta_0 = scipy.stats.uniform.rvs(size = int(1e4))*2*eta
    eff_spin_0 =  scipy.stats.uniform.rvs(size = int(1e4))*2*eff_spin

    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (eff_spin > 0.01) & (eff_spin < 0.99)

    # Appending an array of initial points spread as a uniform rv around the weighted average of the injected values

    mchirp_arr = np.append(mchirp, np.random.choice(mchirp_0[idx], int(N_iter)-1))
    eta_arr = np.append(eta, np.random.choice(eta_0[idx], int(N_iter)-1))
    eff_spin_arr = np.append(eff_spin, np.random.choice(eff_spin_0[idx], int(N_iter)-1))

    # Initializing a function (returning log(mismatch)) for minimizing, with parameters weighted over the SNRs

    injection_parameters = injection_parameters_a.copy()
    for key in injection_parameters:
        injection_parameters[key] = (injection_parameters_a[key]*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+injection_parameters_b[key]*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    fun_mismatch = lambda x: mismatch(h, x[0], x[1], x[2], injection_parameters, det, psd)
    
    # Minimizing the mismatch over all the initial set of points    

    log_mismatch, params = [], []
    for i in range(int(N_iter)):
        res = scipy.optimize.minimize(fun_mismatch, (mchirp_arr[i], eta_arr[i], eff_spin_arr[i]), method='Nelder-Mead', options={'adaptive':True, 'return_all': True}) 
        log_mismatch.append(res.fun)
        params.append(res.allvecs)
    match = 1 - np.power(10, log_mismatch)

    return log_mismatch, match, params

# Importing the waveform metadata and setting the intitial parameters

files = os.listdir('git_overlap/src/output/match_population/match_population_equal_spins/injections')

file_indices_a, file_indices_b = [], []
for file in files:
    if file.endswith('.pkl') and 'GW Waveform A Meta Data' in file:
        index_start = file.find('GW Waveform A Meta Data') + len('GW Waveform A Meta Data ')
        index_end = file.find('.pkl')
        file_index = file[index_start:index_end]
        file_indices_a.append(int(file_index))
    if file.endswith('.pkl') and 'GW Waveform B Meta Data' in file:
        index_start = file.find('GW Waveform B Meta Data') + len('GW Waveform B Meta Data ')
        index_end = file.find('.pkl')
        file_index = file[index_start:index_end]
        file_indices_b.append(int(file_index))

N_sampl = 100
waveform_metadata_a, waveform_metadata_b = [], []
indices = np.random.choice(file_indices_a, N_sampl, replace=False)

keys = ['mass_1_source', 'mass_ratio', 'a_1', 'a_2', 'redshift', 'cos_tilt_1', 'cos_tilt_2', 'phi_12', 'phi_jl', 'cos_theta_jn', 'ra', 'dec', 'psi', 'phase', 'incl', 'cos_theta_zn', 'mass_1', 'mass_2', 'luminosity_distance', 'tilt_1', 'tilt_2', 'theta_jn', 'theta_zn', 'geocent_time', 'snr_det']

data_a = {key: np.zeros(N_sampl) for key in keys}
data_b = {key: np.zeros(N_sampl) for key in keys}

injection_parameters_a, injection_parameters_b = [], []
for i in range(N_sampl):
    params_a, params_b = {}, {}
    for key in data_a:
        params_a[key] = data_a[key][i]
        params_b[key] = data_a[key][i]        
    injection_parameters_a.append(params_a)
    injection_parameters_b.append(params_b)
    
for k in range(N_sampl):
    
    waveform_metadata_a.append(pickle.load(open('git_overlap/src/output/match_population/match_population_equal_spins/injections/GW Waveform A Meta Data %s.pkl' % indices[k], 'rb')))   # Importing Waveform Meta Data
    for key, val in waveform_metadata_a[k]['H1']['parameters'].items():   # Setting the variables
        if key in data_a:
            data_a[key][k] = val

    waveform_metadata_b.append(pickle.load(open('git_overlap/src/output/match_population/match_population_equal_spins/injections/GW Waveform B Meta Data %s.pkl' % indices[k], 'rb')))   # Importing Waveform Meta Data
    for key, val in waveform_metadata_b[k]['H1']['parameters'].items():   # Setting the variables
        if key in data_b:
            data_b[key][k] = val

mchirp_a, mchirp_b = np.power(data_a['mass_1']*data_a['mass_2'], (3/5))/np.power(data_a['mass_1']+data_a['mass_2'], (1/5)), np.power(data_b['mass_1']*data_b['mass_2'], (3/5))/np.power(data_b['mass_1']+data_b['mass_2'], (1/5))
eta_a, eta_b = (data_a['mass_1']*data_a['mass_2'])/np.power(data_a['mass_1']+data_a['mass_2'], 2), (data_b['mass_1']*data_b['mass_2'])/np.power(data_b['mass_1']+data_b['mass_2'], 2)
eff_spin_a, eff_spin_b = data_a['a_1'], data_b['a_1']

delta_tc = data_b['geocent_time'] - data_a['geocent_time']
snr_a, snr_b = data_a['snr_det'], data_b['snr_det']

# Evaluating the log(mismatch) values for the PAIRS

mchirpf_snr, etaf_snr, eff_spinf_snr, match_f_snr = np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl), np.zeros(N_sampl)
for k in range(N_sampl):
    log_mismatch_H1, match_H1, params_H1 = match(waveform_metadata_a[k]['H1']['parameters'], waveform_metadata_b[k]['H1']['parameters'], 'H1', int(N_iter), k)

    # Choosing the convergent values from the parameter space

    mchirp_H1 = [[params_H1[i][j][0] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
    eta_H1 = [[params_H1[i][j][1] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
    eff_spin_H1 = [[params_H1[i][j][2] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]

    mchirp_arr_H1 = [mchirp_H1[i][len(mchirp_H1[i])-1] for i in range(N_iter)]
    eta_arr_H1 = [eta_H1[i][len(eta_H1[i])-1] for i in range(N_iter)]
    eff_spin_arr_H1 = [eff_spin_H1[i][len(eff_spin_H1[i])-1] for i in range(N_iter)]

    min_idx, max_idx = np.argmin(log_mismatch_H1), np.argmax(match_H1)
    log_mismatch_f_H1, match_f_H1 = log_mismatch_H1[min_idx], match_H1[max_idx]
    match_H1[np.isneginf(match_H1)]=0
    match_f_snr[k] = match_f_H1

    mchirpf_H1, etaf_H1, eff_spinf_H1 = mchirp_arr_H1[min_idx], eta_arr_H1[min_idx], eff_spin_arr_H1[min_idx]
    mass_1f_H1, mass_2f_H1 = mconv(mchirpf_H1, etaf_H1)
    mchirpf_snr[k], etaf_snr[k], eff_spinf_snr[k] = mchirpf_H1, etaf_H1, eff_spinf_H1
    np.savetxt('git_overlap/src/output/match_population/match_population_equal_spins/outputs/PAIRS(3D) %s.csv'%(k+1), np.column_stack((match_f_H1, mchirpf_H1, etaf_H1, eff_spinf_H1, mchirp_b[k], mchirp_a[k], eta_a[k], eta_b[k], eff_spin_a[k], eff_spin_b[k], snr_b[k], snr_a[k], delta_tc[k])), delimiter=',')

np.savetxt('git_overlap/src/output/match_population/match_population_equal_spins/outputs/output.csv', np.column_stack((match_f_snr, mchirpf_snr, etaf_snr, eff_spinf_snr, mchirp_b, mchirp_a, eta_a, eta_b, eff_spin_a, eff_spin_b, snr_b, snr_a, delta_tc)), delimiter=',')