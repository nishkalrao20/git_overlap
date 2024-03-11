# 3 Dimensional (Chirp mass, Symmetric mass ratio, Eccentricity) Nelder Mead maximization of Match on PAIRS waveforms

import lal
import pycbc
import bilby
import pickle
import pycbc.psd
import pycbc.types
import pycbc.waveform
import pycbc.detector
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lalsimulation
from pycbc.frame import write_frame
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pycbc.waveform.utils import taper_timeseries

import sys
sys.path.append('/home/nishkal.rao/gweat/src/')
import TEOBResumS_utils as ecc_gen

# Setting some constants

delta_f = 1/8
duration = 8
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048
N_iter = int(12)   # Number of iterations of the initial points

injection_parameters_a = {'mass_1': 50, 
                        'mass_2': 40, 
                        'luminosity_distance': 800, 
                        'a_1': 0, 
                        'a_2': 0, 
                        'tilt_1': 0, 
                        'tilt_2': 0, 
                        'theta_jn': 0, 
                        'phi_12': 1.3264568220000164, 
                        'phi_jl': 4.3767796519003026, 
                        'ra': -0.2231247386804483, 
                        'dec': 0.1839772515975745, 
                        'psi': 1.649825290199503, 
                        'phase': 1.3371755312430611, 
                        'incl': 1.4430344336874731, 
                        'geocent_time': 1126259462.4702857}

injection_parameters_b = {'mass_1': 50, 
                        'mass_2': 30, 
                        'luminosity_distance': 600, 
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
                        'geocent_time': 1126259462.4702857+0.25}

# 3 Dimensional (Chirp mass, Symmetric mass ratio, Eccentricity) Nelder Mead maximization of Match on PAIRS waveforms

def mconv(mchirp, eta):
    """Calculates the masses given the chirp mass and symmetric mass ratio."""
    
    try:
        mtotal = mchirp * np.power(eta, -3/5)
        mass_1 = mtotal*(1+np.sqrt(1-4*eta))/2
        mass_2 = mtotal*(1-np.sqrt(1-4*eta))/2

    except:
        mass_1, mass_2 = mchirp * np.power(.25, -3/5)/2, mchirp * np.power(.25, -3/5)/2

    return mass_1, mass_2

def wf_len_mod_start(wf, extra=1, **prms):
    """
    Taken from GWMAT. Function to modify the starting of a WF so that it starts on an integer GPS time (in sec) + add extra length as specified by the user.

    Parameters
    ----------
    wf :  pycbc.types.TimeSeries
        WF whose length is to be modified.
    extra : int, optional
        Extra length to be added in the beginning after making the WF to start from an integer GPS time (in sec). Default = 1.

    Returns
    -------
    pycbc.types.timeseries.TimeSeries
        Modified waveform starting form an integer time.

    """      

    olen = len(wf)   
    diff = wf.sample_times[0]-np.floor(wf.sample_times[0])  
    #nlen = round(olen+sampling_frequency*(extra+diff))
    dlen = round(sampling_frequency*(extra+diff))
    wf_strain = np.concatenate((np.zeros(dlen), wf))
    t0 = wf.sample_times[0]
    dt = wf.delta_t
    n = dlen
    tnn = t0-(n+1)*dt
    wf_stime = np.concatenate((np.arange(t0-dt,tnn,-dt)[::-1], np.array(wf.sample_times)))
    nwf = pycbc.types.TimeSeries(wf_strain, delta_t=wf.delta_t, epoch=wf_stime[0])
    
    return nwf

def wf_len_mod_end(wf, extra=2, **prms): #post_trig_duration
    """
    Taken from GWMAT. Function to modify the end of a WF so that it ends on an integer GPS time (in sec) + add extra length as specified by the user.

    Parameters
    ----------
    wf : pycbc.types.TimeSeries
        WF whose length is to be modified.
    extra : int, optional
        Extra length to be added towards the end after making the WF to end from an integer GPS time (in sec). 
        Default = 2, which makes sure post-trigger duration is of at least 2 seconds.

    Returns
    -------
    pycbc.types.timeseries.TimeSeries
        Modified waveform ending on an integer time.

    """        

    olen = len(wf)   
    dt = abs(wf.sample_times[-1] - wf.sample_times[-2])
    diff = np.ceil(wf.sample_times[-1]) - (wf.sample_times[-1] + dt)   #wf.sample_times[-1]-int(wf.sample_times[-1])  
    nlen = round(olen + sampling_frequency*(extra+diff))
    wf.resize(nlen)
    
    return wf    

def make_len_power_of_2(wf):
    """
    Taken from GWMAT. Function to modify the length of a waveform so that its duration is a power of 2.

    Parameters
    ----------
    wf : pycbc.types.TimeSeries
        WF whose length is to be modified.
        Modified waveform with duration a power of 2.
    Returns
    -------
    pycbc.types.timeseries.TimeSeries
        Returns the waveform with length a power of 2.

    """    

    dur = wf.duration  
    wf.resize( int(round(wf.sample_rate * np.power(2, np.ceil( np.log2( dur ) ) ))) )
    wf = cyclic_time_shift_of_WF(wf, rwrap = wf.duration - dur )
    
    return wf

def cyclic_time_shift_of_WF(wf, rwrap=0.2):
    """
    Taken from GWMAT. Inspired by PyCBC's function pycbc.types.TimeSeries.cyclic_time_shift(), 
        it shifts the data and timestamps in the time domain by a given number of seconds (rwrap). 
        Difference between this and PyCBCs function is that this function preserves the sample rate of the WFs while cyclically rotating, 
        but the time shift cannot be smaller than the intrinsic sample rate of the data, unlike PyCBc's function.
        To just change the time stamps, do ts.start_time += dt.
        Note that data will be cyclically rotated, so if you shift by 2
        seconds, the final 2 seconds of your data will now be at the
        beginning of the data set.

    Parameters
    ----------
    wf : pycbc.types.TimeSeries
        The waveform for cyclic rotation.
    rwrap : float, optional
        Amount of time to shift the vector. Default = 0.2.

    Returns
    -------
    pycbc.types.TimeSeries
        The time shifted time series.

    """        

    # This function does cyclic time shift of a WF.
    # It is similar to PYCBC's "cyclic_time_shift" except for the fact that it also preserves the Sample Rate of the original WF.
    if rwrap is not None and rwrap != 0:
        sn = abs(int(rwrap/wf.delta_t))     # number of elements to be shifted 
        cycles = int(sn/len(wf))

        cyclic_shifted_wf = wf.copy()

        sn_new = sn - int(cycles * len(wf))

        if rwrap > 0:
            epoch = wf.sample_times[0] - sn_new * wf.delta_t
            if sn_new != 0:
                wf_arr = np.array(wf).copy()
                tmp_wf_p1 = wf_arr[-sn_new:]
                tmp_wf_p2 = wf_arr[:-sn_new] 
                shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)
        else:
            epoch = wf.sample_times[sn_new]
            if sn_new != 0:
                wf_arr = np.array(wf).copy()
                tmp_wf_p1 = wf_arr[sn_new:] 
                tmp_wf_p2 = wf_arr[:sn_new]
                shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)  

        for i in range(cycles):        
                epoch = epoch - np.sign(rwrap)*wf.duration
                wf_arr = np.array(cyclic_shifted_wf)[:]
                cyclic_shifted_wf = pycbc.types.TimeSeries(wf_arr, delta_t = wf.delta_t, epoch = epoch)

        assert len(cyclic_shifted_wf) == len(wf), 'Length mismatch: cyclic time shift added extra length to WF.'
        return cyclic_shifted_wf
    else:
        return wf  

def jframe_to_l0frame(mass_1, mass_2, f_ref, phi_ref=0., theta_jn=0., phi_jl=0., a_1=0., a_2=0., tilt_1=0., tilt_2=0., phi_12=0., **kwargs):  
    """
    [Inherited from PyCBC and lalsimulation.]
        Function to convert J-frame coordinates (which Bilby uses for PE) to L0-frame coordinates (that Pycbc uses for waveform generation).
        J stands for the total angular momentum while L0 stands for the orbital angular momentum.
    """ 

    inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
        lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
            a_1, a_2, mass_1*lal.MSUN_SI, mass_2*lal.MSUN_SI, f_ref,
            phi_ref)
    out_dict = {'inclination': inclination,
                'spin1x': spin1x,
                'spin1y': spin1y,
                'spin1z': spin1z,
                'spin2x': spin2x,
                'spin2y': spin2y,
                'spin2z': spin2z}
    return out_dict

def inject_wf_ecc(injection_parameters, e):
    """
    Generate PyCBC time domain eccentric waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
    """

    lframe = jframe_to_l0frame(mass_1=injection_parameters['mass_1'], 
                               mass_2=injection_parameters['mass_2'], 
                               f_ref=reference_frequency, 
                               theta_jn=injection_parameters['theta_jn'], 
                               phi_jl=injection_parameters['phi_jl'], 
                               a_1=injection_parameters['a_1'], 
                               a_2=injection_parameters['a_2'], 
                               tilt_1=injection_parameters['tilt_1'], 
                               tilt_2=injection_parameters['tilt_2'], 
                               phi_12=injection_parameters['phi_12'])    

    waveform_params = {
        'approximant': 'IMRPhenomXPHM',
        'mass_1': injection_parameters['mass_1'],
        'mass_2': injection_parameters['mass_2'],
        'spin1x': lframe['spin1x'],
        'spin1y': lframe['spin1y'],
        'spin1z': lframe['spin1z'],
        'spin2x': lframe['spin2x'],
        'spin2y': lframe['spin2y'],
        'spin2z': lframe['spin2z'],
        'luminosity_distance': injection_parameters['luminosity_distance'],
        'inclination': lframe['inclination'],
        'coa_phase': injection_parameters['phase'],
        'f_lower': minimum_frequency,
        'f_ref': reference_frequency,
        'delta_t': 1 / sampling_frequency,
        'ecc': e
    }

    pars = ecc_gen.teobresums_pars_update(waveform_params)
    h = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
    hp, hc = pycbc.types.TimeSeries(h['hp'], delta_t = h['hp'].delta_t), pycbc.types.TimeSeries(h['hc'], delta_t = h['hc'].delta_t)
    hp.start_time += injection_parameters['geocent_time']
    hc.start_time += injection_parameters['geocent_time']
        
    det, ifo_signal = dict(), dict()
    for ifo in ['H1', 'L1', 'V1']:
        det[ifo] = pycbc.detector.Detector(ifo)
        ifo_signal[ifo] = det[ifo].project_wave(hp, hc, injection_parameters['ra'], injection_parameters['dec'], injection_parameters['psi'])
        ifo_signal[ifo] = taper_timeseries(ifo_signal[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
        ifo_signal[ifo] = make_len_power_of_2(wf_len_mod_end(wf_len_mod_start(ifo_signal[ifo])))

    ht_H1, ht_L1, ht_V1 = ifo_signal['H1'], ifo_signal['L1'], ifo_signal['V1']

    ht_H1.start_time += injection_parameters['geocent_time']-ht_H1.sample_times[np.argmax(ht_H1)]
    ht_L1.start_time += injection_parameters['geocent_time']-ht_L1.sample_times[np.argmax(ht_L1)]
    ht_V1.start_time += injection_parameters['geocent_time']-ht_V1.sample_times[np.argmax(ht_V1)]
    
    ht = {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

    return ht

def inject_pairs(injection_parameters_a, injection_parameters_b):
    """
    Generate PyCBC time domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
    """

    injection_parameters_c = injection_parameters_b.copy()
    injection_parameters_c['geocent_time'] = injection_parameters_a['geocent_time']

    ht_a, ht_b = inject_wf_ecc(injection_parameters_a, 0), inject_wf_ecc(injection_parameters_c, 0)

    ht = {}
    for det in ht_a.keys():
        ht_a[det].resize(max(len(ht_a[det]), len(ht_b[det])))
        ht_b[det].resize(max(len(ht_a[det]), len(ht_b[det])))

        ht_c = np.roll(ht_b[det]._data, int((injection_parameters_b['geocent_time']-injection_parameters_a['geocent_time'])/(ht_b[det].delta_t)-(np.argmax(ht_b[det])-np.argmax(ht_a[det]))))
        ht_c = pycbc.types.TimeSeries(ht_c, delta_t = ht_b[det].delta_t)  
        ht_c.start_time = ht_b[det].start_time

        ht_pairs = ht_a[det]._data + ht_c._data
        ht_pairs = pycbc.types.TimeSeries(ht_pairs, delta_t = ht_a[det].delta_t)  
        ht_pairs.start_time = ht_a[det].start_time

        ht[det] = ht_pairs

    return ht

def read_psd(det, flen, low_frequency_cutoff=minimum_frequency, delta_f=delta_f):
    """
    Reading the PSD files for the detectors
    """

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/psd_aLIGO_O4high.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=False)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/psd_aLIGO_O4high.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=False)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/psd_aVirgo_O4high_NEW.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=False)

    for i in range(len(psd)):
        if psd[i]==0:
            psd[i]=1e-52 

    return psd

def log_mismatch_calc(h, mchirp, eta, e, injection_parameters, det):
    """Calculating the match defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1 > 100 or mass_2 > 100 or e<0. or e>0.9:
        log_mismatch = 1e6
    
    else:
        # Set the injection parameters and generate the template waveforms

        injection_parameters['mass_1'], injection_parameters['mass_2'] = mass_1, mass_2
        injection_parameters['mass_ratio'] = mass_2/mass_1

        h_templ = inject_wf_ecc(injection_parameters, e)[det] 
        
        # Resize the templates to the length of the waveform and interpolate the PSD
        
        h.resize(max(len(h), len(h_templ)))
        h_templ.resize(max(len(h), len(h_templ)))
        psd = read_psd(det, max(len(h), len(h_templ))//2+1)

        hf, hf_templ = h.to_frequencyseries(), h_templ.to_frequencyseries()

        # Evaluating the matchedfilter match through pycbc

        match = pycbc.filter.matchedfilter.match(hf, hf_templ, low_frequency_cutoff=minimum_frequency, psd=psd, high_frequency_cutoff=maximum_frequency)[0]
        log_mismatch = np.log10(1-match)   # Defining the log(mismatch)
    
    return log_mismatch 

def match(injection_parameters_a, injection_parameters_b, det, N_iter):
    """Initializing the Mismatch function over an iniitial array of uniformily distributed variables around the injected parameters"""

    mass_1_a, mass_2_a = injection_parameters_a['mass_1'], injection_parameters_a['mass_2']
    mass_1_b, mass_2_b = injection_parameters_b['mass_1'], injection_parameters_b['mass_2']

    mchirp_a, mchirp_b = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5)
    eta_a, eta_b = (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)
    mchirp, eta = (mchirp_a+mchirp_b)/2, (eta_a+eta_b)/2
    
    # Generating the PAIRS waveforms and Interpolating the PSD
    
    h = inject_pairs(injection_parameters_a, injection_parameters_b)[det]

    # Distributing the initial array of points as an uniform continuous rv around the weighted average of the injected values

    mchirp_0 = sp.stats.uniform.rvs(size = int(1e4))*2*mchirp
    eta_0 = sp.stats.uniform.rvs(size = int(1e4))*2*eta
    e_0 = sp.stats.uniform.rvs(0, 1, size = int(1e4))
    
    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (0<e_0) & (e_0<0.9)

    # Appending an array of initial points spread as a uniform rv around the weighted average of the injected values

    mchirp_arr = np.append(mchirp, np.random.choice(mchirp_0[idx], int(N_iter)-1))
    eta_arr = np.append(eta, np.random.choice(eta_0[idx], int(N_iter)-1))
    e_arr = np.append(np.mean(e_0[idx]), np.random.choice(e_0[idx], int(N_iter)-1))

    # Initializing a function (returning log(mismatch)) for minimizing, with parameters weighted over the SNRs

    injection_parameters = injection_parameters_a.copy()
    for key in injection_parameters:
        injection_parameters[key] = (injection_parameters_a[key]+injection_parameters_b[key])/2
    fun_mismatch = lambda x: log_mismatch_calc(h, x[0], x[1], x[2], injection_parameters, det)
    
    # Minimizing the mismatch over all the initial set of points    

    log_mismatch, params = [], []
    for i in range(int(N_iter)):
        res = sp.optimize.minimize(fun_mismatch, (mchirp_arr[i], eta_arr[i], e_arr[i]), method='Nelder-Mead', options={'adaptive':True, 'return_all': True}, tol=1e-4) 
        log_mismatch.append(res.fun)
        params.append(res.allvecs)
    match = 1 - np.power(10, log_mismatch)

    return log_mismatch, match, params

mass_1_a, mass_2_a = injection_parameters_a['mass_1'], injection_parameters_a['mass_2']
mass_1_b, mass_2_b = injection_parameters_b['mass_1'], injection_parameters_b['mass_2']

mchirp_a, eta_a = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2)

mchirp_b, eta_b = np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)

# Evaluating the log(mismatch) values for the PAIRS

log_mismatch_H1, match_H1, params_H1 = match(injection_parameters_a, injection_parameters_b, 'H1', int(N_iter))

# Choosing the convergent values from the parameter space

mchirp_H1 = [[params_H1[i][j][0] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
eta_H1 = [[params_H1[i][j][1] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]
e_H1 = [[params_H1[i][j][2] for j in range(len(params_H1[i]))] for i in range(int(N_iter))]

mchirp_arr_H1 = [mchirp_H1[i][len(mchirp_H1[i])-1] for i in range(N_iter)]
eta_arr_H1 = [eta_H1[i][len(eta_H1[i])-1] for i in range(N_iter)]
e_arr_H1 = [e_H1[i][len(e_H1[i])-1] for i in range(N_iter)]

min_idx, max_idx = np.argmin(log_mismatch_H1), np.argmax(match_H1)
log_mismatch_f_H1, match_f_H1 = log_mismatch_H1[min_idx], match_H1[max_idx]
match_H1[np.isneginf(match_H1)]=0

mchirpf_H1, etaf_H1, ef_H1 = mchirp_arr_H1[min_idx], eta_arr_H1[min_idx], e_arr_H1[min_idx]
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
plt.suptitle('Convergence of Chirp Mass $\\mathcal{M}$, Sym Mass Ratio $\\eta$, Eccentricity $e$')

cmap = plt.get_cmap('plasma')
normalize = plt.Normalize(vmin=match_H1.min(), vmax=match_H1.max())

ax[1, 1].set_frame_on(False)
ax[1, 1].axis('off')
ax_3d = fig.add_subplot(2, 2, 4, projection='3d')

for i in range(N_iter):
    mc12_c = ax[0, 0].plot(mchirp_H1[i], c=cmap(normalize(match_H1[i])))
    et12_c = ax[1, 0].plot(eta_H1[i], c=cmap(normalize(match_H1[i])))
    e12_c = ax[0, 1].plot(e_H1[i], c=cmap(normalize(match_H1[i])))
sc = ax_3d.scatter(mchirp_arr_H1, eta_arr_H1, e_arr_H1, c=match_H1, cmap='plasma')

ax[0, 0].set_title('Chirp Mass $\\mathcal{M}$')
ax[0, 0].axhline(y=mchirp_a, color='r', linestyle='--', label='$\\mathcal{M}_A$')
ax[0, 0].axhline(y=mchirp_b, color='r', linestyle='--', label='$\\mathcal{M}_B$')

ax[1, 0].set_title('Sym Mass Ratio $\\eta$')
ax[1, 0].axhline(y=eta_a, color='r', linestyle='--', label='$\\eta_A$')
ax[1, 0].axhline(y=eta_b, color='r', linestyle='--', label='$\\eta_B$')

ax[0, 1].set_title('Eccentricity $e$')
ax_3d.set_title('Chirp Mass $\\mathcal{M}$, Sym Mass Ratio $\\eta$, Eccentricity $e$')
ax_3d.set_xlabel('$\\mathcal{M}$')
ax_3d.set_ylabel('$\\eta$')
ax_3d.set_zlabel('$e$')

ax_3d.scatter(mchirpf_H1, etaf_H1, ef_H1, facecolors='None', edgecolors='black')
ax_3d.text(mchirpf_H1, etaf_H1, ef_H1, '$\star$', verticalalignment='bottom', horizontalalignment='right')

divider1 = make_axes_locatable(ax[0, 1])
cax1 = divider1.append_axes("right", size="5%", pad=0.35)
cbar1 = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), cax=cax1)
cbar1.set_label('$Match$')

divider2 = make_axes_locatable(ax[1, 1])
cax2 = divider2.append_axes("right", size="5%", pad=0.35)
cbar2 = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), cax=cax2)
cbar2.set_label('$Match$')

fig.tight_layout()
plt.savefig('git_overlap/src/output/match_pe/MATCH_ECC.png')
plt.close()

# Printing the required results

print('Injected - SINGLES A: \n %s'%(injection_parameters_a))
print('Injected - SINGLES B: \n %s'%(injection_parameters_b))
print('Recovered (Eccentric): Chirp Mass: %s, Sym. Ratio: %s, Mass1: %s, Mass2: %s; Eccentricity: %s; Match: %s'%(np.round(mchirpf_H1, 3), np.round(etaf_H1, 3), np.round(mass_1f_H1, 3), np.round(mass_2f_H1, 3), np.round(ef_H1, 3), np.round(match_f_H1, 3)))

ht = inject_pairs(injection_parameters_a, injection_parameters_b)

injection_parameters_0 = injection_parameters_a.copy()
for key in injection_parameters_0:
    injection_parameters_0[key] = (injection_parameters_a[key]+injection_parameters_b[key])/2

injection_parameters_0['mass_1'], injection_parameters_0['mass_2'] = mass_1f_H1, mass_2f_H1
injection_parameters_0['geocent_time'] = injection_parameters_a['geocent_time']
injection_parameters_0['eccentricity'] = ef_H1

ht_ecc = inject_wf_ecc(injection_parameters_0, injection_parameters_0['eccentricity'])

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], 'k--', linewidth=1, label='$\\rm{Injected}$')
ax.plot(ht_ecc['H1'].sample_times, ht_ecc['H1'], 'm-', linewidth=1, label='$\\rm{Recovered}$')
ax.set_xlabel('Time $[s]$')
ax.set_ylabel('Strain $h$')
ax.set_xlim(injection_parameters_a['geocent_time']-0.6, injection_parameters_a['geocent_time']+0.2)
ax.legend()
ax.grid(True)
plt.title('Recovered Eccentric Waveforms: Match: %s'%(np.round(match_f_H1, 3)))
plt.savefig('git_overlap/src/output/match_pe/WAVEFORMS_ECC.png')
plt.show()
plt.close()
