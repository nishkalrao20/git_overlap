import lal
import pycbc
import bilby
import pickle
import numpy as np 
import scipy as sp 
import lalsimulation
import pycbc.waveform
from pycbc.waveform.utils import taper_timeseries

import sys
sys.path.append('/home/nishkal.rao/gweat/src/')
import TEOBResumS_utils as ecc_gen

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
duration = 16
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

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

waveform_metadata = pickle.load(open('../../output/injections/GW Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
injection_parameters = waveform_metadata['H1']['parameters']

ht = inject_wf_ecc(injection_parameters, 0)
ht_ecc = inject_wf_ecc(injection_parameters, .2)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], 'k--', linewidth=1, label='$\\rm{Non-Eccentric}$')
ax.plot(ht_ecc['H1'].sample_times, ht_ecc['H1'], 'b-', linewidth=1, label='${\\rm{Eccentric}}\ (e=0.2)$')
ax.set_xlabel('Time $[s]$')
ax.set_ylabel('Strain $h$')
ax.set_xlim(injection_parameters['geocent_time']+4.5, injection_parameters['geocent_time']+5.5)
ax.legend()
ax.grid(True)
plt.savefig('../../output/match_ecc/plots/EccentricSignals1.pdf', format='pdf', bbox_inches="tight")
plt.show()
plt.close()