# 4 Dimensional (Chirp mass, Symmetric mass ratio, Dimensionless Spin magnitudes) Nelder Mead maximization of Match on PAIRS waveforms

import scipy
import pycbc
import bilby
import pickle
import pycbc.psd
import numpy as np
import matplotlib.pyplot as plt

# Setting some constants

delta_f = 1
duration = 100
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048
N_iter = int(3)   # Number of iterations of the initial points

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

def mismatch(h, mchirp, eta, a_1, a_2, injection_parameters, det, psd):
    """Calculating the match/overlap defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or np.isnan(a_1) or np.isnan(a_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800 or a_1 < 0.01 or a_1 > 0.99 or a_2 < 0.01 or a_2 > 0.99:
        log_mismatch = 1e6
    
    else:
        # Set the injection parameters and generate the template waveforms

        injection_parameters['a_1'], injection_parameters['a_2'] = a_1, a_2
        injection_parameters['mass_1'], injection_parameters['mass_2'] = mass_1, mass_2
        injection_parameters['mass_ratio'] = mass_2/mass_1
        injection_parameters['mass_1_source'] = mass_1/(1+injection_parameters['redshift'])

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

def minimize_mismatch(fun_mismatch, mchirp_0, eta_0, a_1_0, a_2_0): 
   """Minimizing the Mismatch function through an adaptive Nelder-Mead optimization"""
   
   res = scipy.optimize.minimize(fun_mismatch, (mchirp_0, eta_0, a_1_0, a_2_0), method='Nelder-Mead', options={'adaptive':True}) 
   
   return res.fun

def match(injection_parameters_a, injection_parameters_b, det, N_iter):
    """Initializing the Mismatch function over an iniitial array of truncated Gaussians around the injected parameters"""

    mass_1_a, mass_2_a = injection_parameters_a['mass_1'], injection_parameters_a['mass_2']
    mass_1_b, mass_2_b = injection_parameters_b['mass_1'], injection_parameters_b['mass_2']
    mchirp_a, mchirp_b = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5)
    eta_a, eta_b = (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)
    mchirp, eta = (mchirp_a+mchirp_b)/2, (eta_a+eta_b)/2
    a_1, a_2 = (injection_parameters_a['a_1']+injection_parameters_b['a_1'])/2, (injection_parameters_a['a_2']+injection_parameters_b['a_2'])/2

    # Generating the PAIRS waveforms and Interpolating the PSD
    
    h, psd = inject_pairs(injection_parameters_a, injection_parameters_b)[det], read_psd(det)

    # Distributing the initial array of points as an uniform continuous rv around the average of the injected values

    mchirp_0 = scipy.stats.uniform.rvs(size = 10*N_iter)*2*mchirp
    eta_0 = scipy.stats.uniform.rvs(size = 10*N_iter)*2*eta
    a_1_0 = scipy.stats.uniform.rvs(size = 10*N_iter)*2*a_1
    a_2_0 = scipy.stats.uniform.rvs(size = 10*N_iter)*2*a_2

    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (a_1_0>0.01) & (a_1_0<0.99) & (a_2_0>0.01) & (a_2_0<0.99)

    # Appending an array of initial points spread as a Gaussian around the injected values

    mchirp_arr = np.append(mchirp, np.random.choice(mchirp_0[idx], N_iter-1))
    eta_arr = np.append(eta, np.random.choice(eta_0[idx], N_iter-1))
    a_1_arr = np.append(a_1, np.random.choice(a_1_0[idx], N_iter-1))
    a_2_arr = np.append(a_2, np.random.choice(a_2_0[idx], N_iter-1))

    # Initializing a function (returning log(mismatch)) for minimizing

    if waveform_metadata_a[det]['optimal_SNR']>waveform_metadata_b[det]['optimal_SNR']:
        fun_mismatch = lambda x: mismatch(h, x[0], x[1], x[2], x[3], injection_parameters_a, det, psd)
    if waveform_metadata_b[det]['optimal_SNR']>waveform_metadata_a[det]['optimal_SNR']:
        fun_mismatch = lambda x: mismatch(h, x[0], x[1], x[2], x[3], injection_parameters_b, det, psd)
    
    # Minimizing the mismatch over all the initial set of points

    log_mismatch = np.vectorize(minimize_mismatch)(fun_mismatch, mchirp_arr, eta_arr, a_1_arr, a_2_arr)
    match = 1 - np.power(10, log_mismatch)

    return log_mismatch, match, mchirp_arr, eta_arr, a_1_arr, a_2_arr

# Importing the waveform metadata and setting the intitial parameters

waveform_metadata_a, waveform_metadata_b = pickle.load(open('git_overlap/src/output/injections/Waveform A Meta Data.pkl', 'rb')), pickle.load(open('git_overlap/src/output/injections/Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data

mass_1_a, mass_2_a, a_1_a, a_2_a = waveform_metadata_a['H1']['parameters']['mass_1'], waveform_metadata_a['H1']['parameters']['mass_2'], waveform_metadata_a['H1']['parameters']['a_1'], waveform_metadata_a['H1']['parameters']['a_2']
mass_1_b, mass_2_b, a_1_b, a_2_b = waveform_metadata_b['H1']['parameters']['mass_1'], waveform_metadata_b['H1']['parameters']['mass_2'], waveform_metadata_b['H1']['parameters']['a_1'], waveform_metadata_b['H1']['parameters']['a_2']

mchirp_a, eta_a = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2)
mchirp_b, eta_b = np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)

# Evaluating the log(mismatch) values for the PAIRS

log_mismatch_H1, match_H1, mchirp_arr_H1, eta_arr_H1, a_1_arr_H1, a_2_arr_H1 = match(waveform_metadata_a['H1']['parameters'], waveform_metadata_b['H1']['parameters'], 'H1', N_iter)

min_idx, max_idx = np.argmin(log_mismatch_H1), np.argmax(match_H1)
log_mismatch_f_H1, match_f_H1 = log_mismatch_H1[min_idx], match_H1[max_idx]
mchirpf_H1, etaf_H1, a_1f_H1, a_2f_H1 = mchirp_arr_H1[min_idx], eta_arr_H1[min_idx], a_1_arr_H1[min_idx], a_2_arr_H1[min_idx]
mass_1f_H1, mass_2f_H1 = mconv(mchirpf_H1, etaf_H1)

# Matplotlib rcParams

plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

# Plotting the iterations with the recovered parameters over a cmap of match values

fig, ax = plt.subplots(1,2, figsize=(13, 6))
plt.suptitle('\\textbf{PAIRS: MATCH}')

# Mchirp - Eta
mc12_c = ax[0].scatter(mchirp_arr_H1, eta_arr_H1, c = match_H1, cmap='plasma')
ax[0].scatter(mchirp_a, eta_a, facecolors = 'None', edgecolors='black')
ax[0].scatter(mchirp_b, eta_b, facecolors = 'None', edgecolors='black')
ax[0].scatter(mchirpf_H1, etaf_H1, facecolors = 'None', edgecolors='black')
ax[0].text(mchirp_a, eta_a,'A', verticalalignment='top', horizontalalignment='left')
ax[0].text(mchirp_b, eta_b,'B', verticalalignment='top', horizontalalignment='left')
ax[0].text(mchirpf_H1, etaf_H1,'F', verticalalignment='bottom', horizontalalignment='right')
ax[0].set_xlabel('$\mathcal{M}$')
ax[0].set_ylabel('$\eta$')
ax[0].set_title('Mass Ratio $\eta$ $-$ Chirp Mass $\mathcal{M}$')
clb1 = plt.colorbar(mc12_c, ax=ax[0])
clb1.ax.set_title('$Match$')

# a_1 - a_2
s12_c = ax[1].scatter(a_1_arr_H1, a_2_arr_H1, c = match_H1, cmap='plasma')
ax[1].scatter(a_1_a, a_2_a, facecolors = 'None', edgecolors='black')
ax[1].scatter(a_1_b, a_2_b, facecolors = 'None', edgecolors='black')
ax[1].scatter(a_1f_H1, a_2f_H1, facecolors = 'None', edgecolors='black')
ax[1].text(a_1_a, a_2_a,'A', verticalalignment='top', horizontalalignment='left')
ax[1].text(a_1_b, a_2_b,'B', verticalalignment='top', horizontalalignment='left')
ax[1].text(a_1f_H1, a_2f_H1,'F', verticalalignment='bottom', horizontalalignment='right')
ax[1].set_xlabel('$a_1$')
ax[1].set_ylabel('$a_2$')
ax[1].set_title('Dimensionless Spin magnitudes: $a_1-a_2$')
clb2 = plt.colorbar(s12_c, ax=ax[1])
clb2.ax.set_title('$Match$')

fig.tight_layout()
plt.savefig('git_overlap/src/output/match/PAIRS.png')
plt.close()

# Printing the required results

print('Injected - SINGLES A: Chirp Mass: %s, Sym. Ratio: %s, a_1: %s, a_2: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_a, 3), np.round(eta_a, 3), np.round(a_1_a, 3), np.round(a_2_a, 3), np.round(mass_1_a, 3), np.round(mass_2_a, 3)))
print('Injected - SINGLES B: Chirp Mass: %s, Sym. Ratio: %s, a_1: %s, a_2: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_b, 3), np.round(eta_b, 3), np.round(a_1_b, 3), np.round(a_2_b, 3), np.round(mass_1_b, 3), np.round(mass_2_b, 3)))
print('Recovered - PAIRS: Chirp Mass: %s, Sym. Ratio: %s, a_1: %s, a_2: %s, Mass1: %s, Mass2: %s; Match: %s'%(np.round(mchirpf_H1, 3), np.round(etaf_H1, 3), np.round(a_1f_H1, 3), np.round(a_2f_H1, 3), np.round(mass_1f_H1, 3), np.round(mass_2f_H1, 3), np.round(match_f_H1, 3)))