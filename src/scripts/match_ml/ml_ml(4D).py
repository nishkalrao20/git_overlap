# 4 Dimensional (Chirp mass, Symmetric mass ratio, Redshifted Lens Mass, Impact Parameter) Nelder Mead maximization of Match on PAIRS waveforms

import pycbc
import bilby
import pickle
import pycbc.psd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append("GWMAT/pnt_Ff_lookup_table/src/cythonized_pnt_lens_class")   
import cythonized_pnt_lens_class as pnt_lens_cy 

# Setting some constants

delta_f = 1
duration = 8
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

    ifos.inject_signal(waveform_generator = waveform_generator, parameters = injection_parameters, raise_error=False)  
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
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency+1, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

def mismatch(h, mchirp, eta, Ml_z, y, injection_parameters, det, psd):
    """Calculating the match/overlap defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800 or y<0.01 or y>5.00 or Ml_z<10 or Ml_z>1e5:
        log_mismatch = 1e6
    
    else:
        # Set the injection parameters and generate the template waveforms

        injection_parameters['mass_1'], injection_parameters['mass_2'] = mass_1, mass_2
        injection_parameters['m_lens'], injection_parameters['y_lens'] = Ml_z, y

        h_templ = inject_wf_lens(injection_parameters, injection_parameters['m_lens'], injection_parameters['y_lens'])[det] 
        
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
    """Initializing the Mismatch function over an iniitial array of uniformily distributed variables around the injected parameters"""

    mass_1, mass_2 = injection_parameters['mass_1'], injection_parameters['mass_2']

    mchirp = np.power(mass_1*mass_2, 3/5)/np.power(mass_1+mass_2, 1/5)
    eta = (mass_1*mass_2)/np.power(mass_1+mass_2, 2)
    
    # Generating the LENSED waveforms and Interpolating the PSD
    
    h, psd = inject_wf_lens(injection_parameters, injection_parameters['m_lens'], injection_parameters['y_lens'])[det], read_psd(det)

    # Distributing the initial array of points as an uniform continuous rv around the weighted average of the injected values

    mchirp_0 = sp.stats.uniform.rvs(size = int(1e4))*2*mchirp
    eta_0 = sp.stats.uniform.rvs(size = int(1e4))*2*eta
    Ml_z_0 = sp.stats.loguniform.rvs(1e2, 1e4, size=int(1e4))
    y_0 = sp.stats.uniform.rvs(0.01, 2, size = int(1e4))
    
    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (0.01<y_0) & (y_0<5.00)

    # Appending an array of initial points spread as a uniform rv around the weighted average of the injected values

    mchirp_arr = np.append(mchirp, np.random.choice(mchirp_0[idx], int(N_iter)-1))
    eta_arr = np.append(eta, np.random.choice(eta_0[idx], int(N_iter)-1))
    Ml_z_arr = np.append(np.mean(Ml_z_0), np.random.choice(Ml_z_0[idx], int(N_iter)-1))
    y_arr = np.append(np.mean(y_0), np.random.choice(y_0[idx], int(N_iter)-1))

    # Initializing a function (returning log(mismatch)) for minimizing, with parameters weighted over the SNRs

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

injection_parameters_0 = {'mass_1': 10, 
                        'mass_2': 50, 
                        'luminosity_distance': 800, 
                        'a_1': 0, 
                        'a_2': 0, 
                        'tilt_1': 0, 
                        'tilt_2': 0, 
                        'theta_jn': 0, 
                        'phi_12': 2.3264568220000164, 
                        'phi_jl': 2.3767796519003026, 
                        'ra': 0.8231247386804483, 
                        'dec': -0.7839772515975745, 
                        'psi': 2.649825290199503, 
                        'phase': 0.3371755312430611, 
                        'incl': 0.4430344336874731, 
                        'geocent_time': 1126259462.4702857,
                        'm_lens': 1e2,
                        'y_lens': 1}

mass_1, mass_2 = injection_parameters_0['mass_1'], injection_parameters_0['mass_2']
Ml_z0, y0 = injection_parameters_0['m_lens'], injection_parameters_0['y_lens']
mchirp, eta = np.power(mass_1*mass_2, 3/5)/np.power(mass_1+mass_2, 1/5), (mass_1*mass_2)/np.power(mass_1+mass_2, 2)

# Evaluating the log(mismatch) values for the PAIRS

injection_parameters = injection_parameters_0.copy()
log_mismatch_H1, match_H1, params_H1 = match(injection_parameters, 'H1', int(N_iter))

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

# Matplotlib rcParams

plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

# Plotting the convergence of the parameters, and the iterations with the recovered parameters over a cmap of match values

fig, ax = plt.subplots(2, 2, figsize=(18, 16))
plt.suptitle('LENSEDvsLENSED: Convergence of Chirp Mass $\\mathcal{M}$, Sym Mass Ratio $\\eta$, Redshifted Lens Mass $M_{\\ell}^z$, and Impact Parameter $y$')

cmap = plt.get_cmap('plasma')
normalize = plt.Normalize(vmin=match_H1.min(), vmax=match_H1.max())
for i in range(N_iter):
    mc12_c = ax[0, 0].plot(mchirp_H1[i], c=cmap(normalize(match_H1[i])))
    et12_c = ax[1, 0].plot(eta_H1[i], c=cmap(normalize(match_H1[i])))
    ml12_c = ax[0, 1].plot(Ml_z_H1[i], c=cmap(normalize(match_H1[i])))
    y12_c = ax[1, 1].plot(y_H1[i], c=cmap(normalize(match_H1[i])))

ax[0, 0].set_title('Chirp Mass $\\mathcal{M}$')
ax[0, 0].axhline(y=mchirp, color='r', linestyle='--', label='$\\mathcal{M}$')

ax[1, 0].set_title('Sym Mass Ratio $\\eta$')
ax[1, 0].axhline(y=eta, color='r', linestyle='--', label='$\\eta$')

ax[0, 1].set_title('Redshifted Lens Mass $M_{\ell}^z$')
ax[0, 1].axhline(y=Ml_z0, color='r', linestyle='--', label='$M_{\\ell}^z$')

ax[1, 1].set_title('Impact Parameter $y$')
ax[1, 1].axhline(y=y0, color='r', linestyle='--', label='$y$')

divider1 = make_axes_locatable(ax[0, 1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), cax=cax1)
cbar1.set_label('$Match$')

divider2 = make_axes_locatable(ax[1, 1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), cax=cax2)
cbar2.set_label('$Match$')

fig.tight_layout()
plt.savefig('git_overlap/src/output/match_ml/ml_ml/LENSED(4D).png')
plt.close()

# Printing the required results

print('Injected - LENSED: \n %s'%(injection_parameters_0))
print('Recovered - LENSED: Chirp Mass: %s, Sym. Ratio: %s, Mass1: %s, Mass2: %s; Redshifted Lens Mass: %s, Impact Parameter: %s; Match: %s'%(np.round(mchirpf_H1, 3), np.round(etaf_H1, 3), np.round(mass_1f_H1, 3), np.round(mass_2f_H1, 3), np.round(Ml_zf_H1, 3), np.round(yf_H1, 3), np.round(match_f_H1, 3)))

ht = inject_wf_lens(injection_parameters_0, injection_parameters_0['m_lens'], injection_parameters_0['y_lens'])
injection_parameters_0['mass_1'], injection_parameters_0['mass_2'] = mass_1f_H1, mass_2f_H1
injection_parameters_0['m_lens'], injection_parameters_0['y_lens'] = Ml_zf_H1, yf_H1
ht_lens = inject_wf_lens(injection_parameters, injection_parameters['m_lens'], injection_parameters['y_lens'])

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], 'k--', linewidth=1, label='$\\rm{Injected}$')
ax.plot(ht_lens['H1'].sample_times, ht_lens['H1'], 'm-', linewidth=1, label='$\\rm{Recovered}$')
ax.set_xlabel('Time $[s]$')
ax.set_ylabel('Strain $h$')
ax.set_xlim(injection_parameters['geocent_time']-0.7, injection_parameters['geocent_time']+0.1)
ax.legend()
ax.grid(True)
plt.savefig('git_overlap/src/output/match_ml/ml_ml/LENSEDvsLENSED.png')
plt.show()
plt.close()
