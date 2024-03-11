# 2 Dimensional (Chirp mass, Symmetric mass ratio) Nelder Mead maximization of Match on SINGLES waveforms

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
N_iter = int(12)   # Number of iterations of the initial points

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

def read_psd(det):
    """Reading the PSD files for the detectors"""

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

def mismatch(h, mchirp, eta, injection_parameters, det, psd):
    """Calculating the match/overlap defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800:
        log_mismatch = 1e6
    
    else:
        # Set the injection parameters and generate the template waveforms

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

def match(injection_parameters, det, N_iter):
    """Initializing the Mismatch function over an iniitial array of truncated Gaussians around the injected parameters"""

    mass_1, mass_2 = injection_parameters['mass_1'], injection_parameters['mass_2']
    mchirp, eta = np.power(mass_1*mass_2, 3/5)/np.power(mass_1+mass_2, 1/5), (mass_1*mass_2)/np.power(mass_1+mass_2, 2)

    # Generating the PAIRS waveforms and Interpolating the PSD
    
    h, psd = inject_wf(injection_parameters)[det], read_psd(det)

    # Distributing the initial array of points as an truncated normal continuous rv around the injected values

    mchirp_0 = scipy.stats.truncnorm.rvs(-3, 3, size = int(1e4))/2+mchirp
    eta_0 = scipy.stats.truncnorm.rvs(-3, 3, size = int(1e4))/10+eta 

    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25)

    # Appending an array of initial points spread as a Gaussian around the injected values

    mchirp_arr = np.random.choice(mchirp_0[idx], int(N_iter))
    eta_arr = np.random.choice(eta_0[idx], int(N_iter))

    # Initializing a function (returning log(mismatch)) for minimizing, with parameters weighted over the SNRs

    fun_mismatch = lambda x: mismatch(h, x[0], x[1], injection_parameters, det, psd)
    
    # Minimizing the mismatch over all the initial set of points    

    log_mismatch, params = [], []
    for i in range(int(N_iter)):
        res = scipy.optimize.minimize(fun_mismatch, (mchirp_arr[i], eta_arr[i]), method='Nelder-Mead', options={'adaptive':True, 'return_all': True}) 
        log_mismatch.append(res.fun)
        params.append(res.allvecs)
    match = 1 - np.power(10, log_mismatch)

    return log_mismatch, match, params

# Importing the waveform metadata and setting the intitial parameters

waveform_metadata_a, waveform_metadata_b = pickle.load(open('git_overlap/src/output/injections/GW Waveform A Meta Data.pkl', 'rb')), pickle.load(open('git_overlap/src/output/injections/GW Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data

mass_1_a, mass_2_a = waveform_metadata_a['H1']['parameters']['mass_1'], waveform_metadata_a['H1']['parameters']['mass_2']
mass_1_b, mass_2_b = waveform_metadata_b['H1']['parameters']['mass_1'], waveform_metadata_b['H1']['parameters']['mass_2']

mchirp_a, eta_a = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2)
mchirp_b, eta_b = np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)

# Evaluating the log(mismatch) values for SINGLES A

log_mismatch_a_H1, match_a_H1, params_a_H1 = match(waveform_metadata_a['H1']['parameters'], 'H1', int(N_iter))

# Choosing the convergent values from the parameter space

mchirp_a_H1 = [[params_a_H1[i][j][0] for j in range(len(params_a_H1[i]))] for i in range(int(N_iter))]
eta_a_H1 = [[params_a_H1[i][j][1] for j in range(len(params_a_H1[i]))] for i in range(int(N_iter))]

mchirp_arr_a_H1 = [mchirp_a_H1[i][len(mchirp_a_H1[i])-1] for i in range(N_iter)]
eta_arr_a_H1 = [eta_a_H1[i][len(eta_a_H1[i])-1] for i in range(N_iter)]

match_a_H1 = np.nan_to_num(match_a_H1, nan=1, neginf=0)
min_idx_a, max_idx_a = np.argmin(log_mismatch_a_H1), np.argmax(match_a_H1)
log_mismatch_a_f_H1, match_a_f_H1 = log_mismatch_a_H1[min_idx_a], match_a_H1[max_idx_a]

mchirpf_a_H1, etaf_a_H1 = mchirp_arr_a_H1[min_idx_a], eta_arr_a_H1[min_idx_a]
mass_1f_a_H1, mass_2f_a_H1 = mconv(mchirpf_a_H1, etaf_a_H1)

# Evaluating the log(mismatch) values for SINGLES B

log_mismatch_b_H1, match_b_H1, params_b_H1 = match(waveform_metadata_b['H1']['parameters'], 'H1', int(N_iter))

# Choosing the convergent values from the parameter space

mchirp_b_H1 = [[params_b_H1[i][j][0] for j in range(len(params_b_H1[i]))] for i in range(int(N_iter))]
eta_b_H1 = [[params_b_H1[i][j][1] for j in range(len(params_b_H1[i]))] for i in range(int(N_iter))]

mchirp_arr_b_H1 = [mchirp_b_H1[i][len(mchirp_b_H1[i])-1] for i in range(N_iter)]
eta_arr_b_H1 = [eta_b_H1[i][len(eta_b_H1[i])-1] for i in range(N_iter)]

match_b_H1 = np.nan_to_num(match_b_H1, nan=1, neginf=0)
min_idx_b, max_idx_b = np.argmin(log_mismatch_b_H1), np.argmax(match_b_H1)
log_mismatch_b_f_H1, match_b_f_H1 = log_mismatch_b_H1[min_idx_b], match_b_H1[max_idx_b]

mchirpf_b_H1, etaf_b_H1 = mchirp_arr_b_H1[min_idx_b], eta_arr_b_H1[min_idx_b]
mass_1f_b_H1, mass_2f_b_H1 = mconv(mchirpf_b_H1, etaf_b_H1)

# Matplotlib rcParams

plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

# Plotting the convergence of the parameters, and the iterations with the recovered parameters over a cmap of match values

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.suptitle('SINGLES A: Convergence of Chirp Mass $\\mathcal{M}$ and Sym Mass Ratio $\\eta$')

cmap = plt.get_cmap('plasma')
normalize = plt.Normalize(vmin=match_a_H1.min(), vmax=match_a_H1.max())
for i in range(N_iter):
    ax[0].plot(mchirp_a_H1[i], c=cmap(normalize(match_a_H1[i])))
    ax[1].plot(eta_a_H1[i], c=cmap(normalize(match_a_H1[i])))
ax[0].set_title('Chirp Mass $\\mathcal{M}$')
ax[0].axhline(y=mchirp_a, color='r', linestyle='--', label='$\\mathcal{M}_A$')
ax[1].set_title('Sym Mass Ratio $\\eta$') 
ax[1].axhline(y=eta_a, color='r', linestyle='--', label='$\\eta_A$')

mc12_c = ax[2].scatter(mchirp_arr_a_H1, eta_arr_a_H1, c = match_a_H1, cmap='plasma')
ax[2].scatter(mchirp_a, eta_a, facecolors = 'None', edgecolors='black')
ax[2].scatter(mchirpf_a_H1, etaf_a_H1, facecolors = 'None', edgecolors='black')
ax[2].text(mchirp_a, eta_a,'A', verticalalignment='top', horizontalalignment='left')
ax[2].text(mchirpf_a_H1, etaf_a_H1,'F', verticalalignment='bottom', horizontalalignment='right')
ax[2].set_xlabel('$\mathcal{M}$')
ax[2].set_ylabel('$\eta$')
ax[2].set_title('Mass Ratio $\\eta$ $-$ Chirp Mass $\\mathcal{M}$')
clb1 = plt.colorbar(mc12_c, ax=ax[2])
clb1.ax.set_title('$Match$')

fig.tight_layout()
plt.savefig('git_overlap/src/output/match/SINGLES:A(2D).png')
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.suptitle('SINGLES B: Convergence of Chirp Mass $\\mathcal{M}$ and Sym Mass Ratio $\\eta$')

cmap = plt.get_cmap('plasma')
normalize = plt.Normalize(vmin=match_b_H1.min(), vmax=match_b_H1.max())
for i in range(N_iter):
    ax[0].plot(mchirp_b_H1[i], c=cmap(normalize(match_b_H1[i])))
    ax[1].plot(eta_b_H1[i], c=cmap(normalize(match_b_H1[i])))
ax[0].set_title('Chirp Mass $\\mathcal{M}$')
ax[0].axhline(y=mchirp_b, color='r', linestyle='--', label='$\\mathcal{M}_B$')
ax[1].set_title('Sym Mass Ratio $\\eta$') 
ax[1].axhline(y=eta_b, color='r', linestyle='--', label='$\\eta_B$')

mc12_c = ax[2].scatter(mchirp_arr_b_H1, eta_arr_b_H1, c = match_b_H1, cmap='plasma')
ax[2].scatter(mchirp_b, eta_b, facecolors = 'None', edgecolors='black')
ax[2].scatter(mchirpf_b_H1, etaf_b_H1, facecolors = 'None', edgecolors='black')
ax[2].text(mchirp_b, eta_b,'B', verticalalignment='top', horizontalalignment='left')
ax[2].text(mchirpf_b_H1, etaf_b_H1,'F', verticalalignment='bottom', horizontalalignment='right')
ax[2].set_xlabel('$\mathcal{M}$')
ax[2].set_ylabel('$\eta$')
ax[2].set_title('Mass Ratio $\\eta$ $-$ Chirp Mass $\\mathcal{M}$')
clb1 = plt.colorbar(mc12_c, ax=ax[2])
clb1.ax.set_title('$Match$')

fig.tight_layout()
plt.savefig('git_overlap/src/output/match/SINGLES:B(2D).png')
plt.close()

# Printing the required results

print('Injected - SINGLES A: GW150914: \n %s'%(waveform_metadata_a['H1']['parameters']))
print('Recovered - SINGLES A: Chirp Mass: %s, Sym. Ratio: %s, Mass1: %s, Mass2: %s; Match: %s'%(np.round(mchirpf_a_H1, 3), np.round(etaf_a_H1, 3), np.round(mass_1f_a_H1, 3), np.round(mass_2f_a_H1, 3), np.round(match_a_f_H1, 3)))
print('Injected - SINGLES B: GW170814: \n %s'%(waveform_metadata_b['H1']['parameters']))
print('Recovered - SINGLES B: Chirp Mass: %s, Sym. Ratio: %s, Mass1: %s, Mass2: %s; Match: %s'%(np.round(mchirpf_b_H1, 3), np.round(etaf_b_H1, 3), np.round(mass_1f_b_H1, 3), np.round(mass_2f_b_H1, 3), np.round(match_b_f_H1, 3)))