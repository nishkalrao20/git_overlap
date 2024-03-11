#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# 3 Dimensional (Chirp mass, Symmetric mass ratio, Eccentricity) Nelder Mead maximization of Match on PAIRS waveforms, by varying SNR of SINGLES_B

import os
import sys

import lal
import pycbc
import bilby
import pickle
import pycbc.psd
import pycbc.types
import pycbc.waveform
import pycbc.detector
import lalsimulation
from pycbc.waveform.utils import taper_timeseries

import numpy as np
import scipy as sp

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
        psd = pycbc.psd.read.from_txt('/home/nishkal.rao/git_overlap/src/psds/psd_aLIGO_O4high.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=False)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('/home/nishkal.rao/git_overlap/src/psds/psd_aLIGO_O4high.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=False)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('/home/nishkal.rao/git_overlap/src/psds/psd_aVirgo_O4high_NEW.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=False)

    for i in range(len(psd)):
        if psd[i]==0:
            psd[i]=1e-52 

    return psd

def log_mismatch_calc(h, mchirp, eta, e, injection_parameters, det):
    """Calculating the match defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if np.isnan(mass_1) or np.isnan(mass_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1 > 100 or mass_2 > 100 or e<0. or e>0.5:
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

        match = pycbc.filter.matchedfilter.match(hf, hf_templ, low_frequency_cutoff=minimum_frequency, psd=None, high_frequency_cutoff=maximum_frequency)[0]
        log_mismatch = np.log10(1-match)   # Defining the log(mismatch)
    
    return log_mismatch 

def match(injection_parameters_a, injection_parameters_b, det, N_iter, k):
    """Initializing the Mismatch function over an iniitial array of uniformily distributed variables around the injected parameters"""

    mass_1_a, mass_2_a = injection_parameters_a['mass_1'], injection_parameters_a['mass_2']
    mass_1_b, mass_2_b = injection_parameters_b['mass_1'], injection_parameters_b['mass_2']

    mchirp_a, mchirp_b = np.power(mass_1_a*mass_2_a, 3/5)/np.power(mass_1_a+mass_2_a, 1/5), np.power(mass_1_b*mass_2_b, 3/5)/np.power(mass_1_b+mass_2_b, 1/5)
    eta_a, eta_b = (mass_1_a*mass_2_a)/np.power(mass_1_a+mass_2_a, 2), (mass_1_b*mass_2_b)/np.power(mass_1_b+mass_2_b, 2)
    
    mchirp = (mchirp_a*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+mchirp_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    eta = (eta_a*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+eta_b*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    
    # Generating the PAIRS waveforms and Interpolating the PSD
    
    h = inject_pairs(injection_parameters_a, injection_parameters_b)[det]

    # Distributing the initial array of points as an uniform continuous rv around the weighted average of the injected values

    mchirp_0 = sp.stats.uniform.rvs(size = int(1e4))*2*mchirp
    eta_0 = sp.stats.uniform.rvs(size = int(1e4))*2*eta
    e_0 = sp.stats.loguniform.rvs(1e-3, 0.5, size = int(1e4))
    
    # Selecting the physically alllowed solutions

    idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (0<e_0) & (e_0<0.5)

    # Appending an array of initial points spread as a uniform rv around the weighted average of the injected values

    mchirp_arr = np.append(mchirp, np.random.choice(mchirp_0[idx], int(N_iter)-1))
    eta_arr = np.append(eta, np.random.choice(eta_0[idx], int(N_iter)-1))
    e_arr = np.append(np.mean(e_0[idx]), np.random.choice(e_0[idx], int(N_iter)-1))

    # Initializing a function (returning log(mismatch)) for minimizing, with parameters weighted over the SNRs

    injection_parameters = injection_parameters_a.copy()
    for key in injection_parameters:
        injection_parameters[key] = (injection_parameters_a[key]*np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+injection_parameters_b[key]*np.abs(waveform_metadata_b[k][det]['optimal_SNR']))/(np.abs(waveform_metadata_a[k][det]['optimal_SNR'])+np.abs(waveform_metadata_b[k][det]['optimal_SNR']))
    fun_mismatch = lambda x: log_mismatch_calc(h, x[0], x[1], x[2], injection_parameters, det)
    
    # Minimizing the mismatch over all the initial set of points    

    log_mismatch, params = [], []
    for i in range(int(N_iter)):
        res = sp.optimize.minimize(fun_mismatch, (mchirp_arr[i], eta_arr[i], e_arr[i]), method='Nelder-Mead', options={'adaptive':True, 'return_all': True}, tol=1e-4) 
        log_mismatch.append(res.fun)
        params.append(res.allvecs)
    match = 1 - np.power(10, log_mismatch)

    return log_mismatch, match, params

# Importing the waveform metadata and setting the intitial parameters

files = os.listdir('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/injections')

file_indices = []
for file in files:
    if file.endswith('.pkl') and 'GW Waveform A Meta Data' in file:
        index_start = file.find('GW Waveform A Meta Data') + len('GW Waveform A Meta Data ')
        index_end = file.find('.pkl')
        file_index = file[index_start:index_end]
        file_indices.append(int(file_index))
N_sampl = len(file_indices)
queue = 500
k = int(sys.argv[1])
indices = file_indices[k*int(N_sampl/queue):(k+1)*int(N_sampl/queue)]

keys = ['mass_1_source', 'mass_ratio', 'a_1', 'a_2', 'redshift', 'cos_tilt_1', 'cos_tilt_2', 'phi_12', 'phi_jl', 'cos_theta_jn', 'ra', 'dec', 'psi', 'phase', 'incl', 'cos_theta_zn', 'mass_1', 'mass_2', 'luminosity_distance', 'tilt_1', 'tilt_2', 'theta_jn', 'theta_zn', 'geocent_time', 'snr_det']

data_a, data_b = {key: np.zeros(len(indices)) for key in keys}, {key: np.zeros(len(indices)) for key in keys}
waveform_metadata_a, waveform_metadata_b = [], []
for k, idx in enumerate(indices):
    
    waveform_metadata_a.append(pickle.load(open('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/injections/GW Waveform A Meta Data %s.pkl' % idx, 'rb')))   # Importing Waveform Meta Data
    for key, val in waveform_metadata_a[k]['H1']['parameters'].items():   # Setting the variables
        if key in data_a:
            data_a[key][k] = val

    waveform_metadata_b.append(pickle.load(open('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/injections/GW Waveform B Meta Data %s.pkl' % idx, 'rb')))   # Importing Waveform Meta Data
    for key, val in waveform_metadata_b[k]['H1']['parameters'].items():   # Setting the variables
        if key in data_b:
            data_b[key][k] = val

mchirp_a, mchirp_b = np.power(data_a['mass_1']*data_a['mass_2'], (3/5))/np.power(data_a['mass_1']+data_a['mass_2'], (1/5)), np.power(data_b['mass_1']*data_b['mass_2'], (3/5))/np.power(data_b['mass_1']+data_b['mass_2'], (1/5))
eta_a, eta_b = (data_a['mass_1']*data_a['mass_2'])/np.power(data_a['mass_1']+data_a['mass_2'], 2), (data_b['mass_1']*data_b['mass_2'])/np.power(data_b['mass_1']+data_b['mass_2'], 2)
eff_spin_a, eff_spin_b = data_a['a_1'], data_b['a_1']

delta_tc = data_b['geocent_time'] - data_a['geocent_time']
snr_a, snr_b = data_a['snr_det'], data_b['snr_det']

# Evaluating the log(mismatch) values for the PAIRS
for k, idx in enumerate(indices):

    log_mismatch_H1, match_H1, params_H1 = match(waveform_metadata_a[k]['H1']['parameters'], waveform_metadata_b[k]['H1']['parameters'], 'H1', int(N_iter), k)

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
    np.savetxt('/home/nishkal.rao/git_overlap/src/output/match_ml_ecc_population/outputs_ecc/PAIRS(3D) %s.csv'%(idx), np.column_stack((match_f_H1, mchirpf_H1, etaf_H1, ef_H1)), delimiter=',')
