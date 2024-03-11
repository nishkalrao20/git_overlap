# https://arxiv.org/pdf/2304.06734.pdf
# Generate BBH Injections (SINGLES/ PAIRS) and inject into detectors (H1, L1, V1) 

import pycbc
import numpy as np
import pycbc.types
import pycbc.waveform
import pycbc.detector
import lalsimulation
import lal
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

def mconv(mchirp, q):
    """Calculates the masses given the chirp mass and mass ratio."""
    
    mass_1 = mchirp*((1+q)**(1/5))/(q**(3/5))
    mass_2 = mass_1*q

    return mass_1, mass_2

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

    det, ifo_signal = dict(), dict()
    for ifo in ['H1', 'L1', 'V1']:
        det[ifo] = pycbc.detector.Detector(ifo)
        ifo_signal[ifo] = det[ifo].project_wave(hp, hc, ra, dec, psi)
        ifo_signal[ifo] = taper_timeseries(ifo_signal[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
        ifo_signal[ifo] = make_len_power_of_2(wf_len_mod_end(wf_len_mod_start(ifo_signal[ifo])))

    ht_H1, ht_L1, ht_V1 = ifo_signal['H1'], ifo_signal['L1'], ifo_signal['V1']

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

    det, ifo_signal_a, ifo_signal_b = dict(), dict(), dict()

    for ifo in ['H1', 'L1', 'V1']:
        det[ifo] = pycbc.detector.Detector(ifo)

        hp_a, hc_a = pycbc.waveform.get_td_waveform(**waveform_params_a)
        hp_a.start_time += start_time_a
        hc_a.start_time += start_time_a

        ifo_signal_a[ifo] = det[ifo].project_wave(hp_a, hc_a, ra_a, dec_a, psi_a)
        ifo_signal_a[ifo] = taper_timeseries(ifo_signal_a[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
        ifo_signal_a[ifo] = make_len_power_of_2(wf_len_mod_end(wf_len_mod_start(ifo_signal_a[ifo])))
        
        hp_b, hc_b = pycbc.waveform.get_td_waveform(**waveform_params_b)
        hp_b.start_time += start_time_b
        hc_b.start_time += start_time_b

        ifo_signal_b[ifo] = det[ifo].project_wave(hp_b, hc_b, ra_b, dec_b, psi_b)
        ifo_signal_b[ifo] = taper_timeseries(ifo_signal_b[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
        ifo_signal_b[ifo] = make_len_power_of_2(wf_len_mod_end(wf_len_mod_start(ifo_signal_b[ifo])))

    ht_H1_a, ht_L1_a, ht_V1_a = ifo_signal_a['H1'], ifo_signal_a['L1'], ifo_signal_a['V1']
    ht_H1_b, ht_L1_b, ht_V1_b = ifo_signal_b['H1'], ifo_signal_b['L1'], ifo_signal_b['V1']

    hf_a_H1, hf_a_L1, hf_a_V1 = ht_H1_a.to_frequencyseries(), ht_L1_a.to_frequencyseries(), ht_V1_a.to_frequencyseries()
    hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_b.to_frequencyseries(), ht_L1_b.to_frequencyseries(), ht_V1_b.to_frequencyseries()

    hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (start_time_b - start_time_a) * hf_b_L1.sample_frequencies)
    hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (start_time_b - start_time_a) * hf_b_L1.sample_frequencies)
    hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (start_time_b - start_time_a) * hf_b_V1.sample_frequencies)

    ht_H1, ht_L1, ht_V1 = hf_H1.to_timeseries(), hf_L1.to_timeseries(), hf_V1.to_timeseries()
    
    return ht_H1, ht_L1, ht_V1

mchirp = np.random.uniform(24.1, 32.1, 10)
for i in range(len(mchirp)):
    mass1_a, mass2_a = mconv(28.1, 0.806)
    mass1_b, mass2_b = mconv(mchirp[i], 0.9)

    lframe_a = jframe_to_l0frame(mass_1=mass1_a, mass_2=mass2_a, f_ref=reference_frequency, theta_jn=2.33101251230555, phi_jl=1.4698141271696548, a_1=0.5473234710877908, a_2=0.27126319787312464, tilt_1=1.4703452023562735, tilt_2=1.4060762787385255, phi_12=1.7952004204194505)    
    lframe_b = jframe_to_l0frame(mass_1=mass1_b, mass_2=mass2_b, f_ref=reference_frequency, theta_jn=2.33101251230555, phi_jl=1.4698141271696548, a_1=0.5473234710877908, a_2=0.27126319787312464, tilt_1=1.4703452023562735, tilt_2=1.4060762787385255, phi_12=1.7952004204194505)    

    ht_H1, ht_L1, ht_V1 = inject_pairs(mass1_a=mass1_a, mass2_a=mass2_a, spin1x_a=lframe_a['spin1x'], spin1y_a=lframe_a['spin1y'], spin1z_a=lframe_a['spin1z'], spin2x_a=lframe_a['spin2x'], spin2y_a=lframe_a['spin2y'], spin2z_a=lframe_a['spin2z'], luminosity_distance_a=402.2372317973936, iota_a=lframe_a['inclination'], phase_a=1.4698141271696548, ra_a=1.4872165580410979, dec_a=-1.2571099207652874, psi_a=1.7952004204194505, start_time_a=1126259462.4116447, mass1_b=mass1_b, mass2_b=mass2_b, spin1x_b=lframe_b['spin1x'], spin1y_b=lframe_b['spin1y'], spin1z_b=lframe_b['spin1z'], spin2x_b=lframe_b['spin2x'], spin2y_b=lframe_b['spin2y'], spin2z_b=lframe_b['spin2z'], luminosity_distance_b=402.2372317973936, iota_b=lframe_b['inclination'], phase_b=1.4698141271696548, ra_b=1.4872165580410979, dec_b=-1.2571099207652874, psi_b=1.7952004204194505, start_time_b=1126259462.4116447+0.025)
    
    write_frame("git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_H1_%s.gwf"%(i+1), "H1:PyCBC_Injection", ht_H1)
    write_frame("git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_L1_%s.gwf"%(i+1), "L1:PyCBC_Injection", ht_L1)
    write_frame("git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_V1_%s.gwf"%(i+1), "V1:PyCBC_Injection", ht_V1)
