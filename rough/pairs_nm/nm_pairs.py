# 4 Dimensional (Chirp mass, Symmetric mass ratio, Spin1z, Spin2z) Nelder Mead simplex optimization on PAIRS

import math
import scipy
import pycbc
import pickle
import pycbc.psd
import numpy as np
import matplotlib
import pycbc.waveform
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'text.usetex' : True})

delta_f = 1
minimum_frequency = 20
maximum_frequency = 1024
sampling_frequency = 2048
delta_t = 1/sampling_frequency

def mconv(mchirp, eta):
    """Calculates the masses given the chirp mass and symmetric mass ratio."""
    
    mtotal = mchirp * np.power(eta, -3/5)
    mass_1 = mtotal*(1+np.sqrt(1-4*eta))/2
    mass_2 = mtotal*(1-np.sqrt(1-4*eta))/2

    if math.isnan(mass_1) or math.isnan(mass_2) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800:
        mass_1, mass_2 = 50, 50   # To prevent NaN errors

    else:  
        mass_1 = mtotal*(1+np.sqrt(1-4*eta))/2
        mass_2 = mtotal*(1-np.sqrt(1-4*eta))/2 
        
    return mass_1, mass_2

def wf(mchirp, eta, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time, det):
    """Generating waveforms and injecting into detectors"""

    mass_1, mass_2 = mconv(mchirp, eta)

    hp, hc = pycbc.waveform.get_td_waveform(approximant='IMRPhenomPv2', mass1=mass_1, mass2=mass_2, distance=luminosity_distance, spin1z=spin1z, spin2z=spin2z, 
                                                            inclination=incl, coa_phase=phase, delta_t=delta_t, f_lower=minimum_frequency, f_final=maximum_frequency)

    hp, hc = hp.cyclic_time_shift(hp.duration-geocent_time), hc.cyclic_time_shift(hc.duration-geocent_time)   # To avoid wrapaaround

    Fp, Fc = pycbc.detector.Detector(det).antenna_pattern(ra, dec, psi, geocent_time)   # Antenna Patterns
    h = Fp*hp + Fc*hc   # h = F+*h+ + Fx*hx

    return h

def pairs_wf(mchirp_a, eta_a, spin1z_a, spin2z_a, luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a,
            mchirp_b, eta_b, spin1z_b, spin2z_b, luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b, 
            det):
    """Generating SINGLES waveforms and injecting into detectors"""

    mass_1_a, mass_2_a = mconv(mchirp_a, eta_a)
    mass_1_b, mass_2_b = mconv(mchirp_b, eta_b)

    # Generating the SINGLES

    h_a = wf(mchirp_a, eta_a, spin1z_a, spin2z_a, luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a, det)
    h_b = wf(mchirp_b, eta_b, spin1z_b, spin2z_b, luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b, det)

    # Padding the arrays to equalize the length
    
    h_b = np.lib.pad(h_b, (int((geocent_time_b - geocent_time_a)/delta_t), 0), mode='constant')
    h_a = np.lib.pad(h_a,(0,len(h_b)-len(h_a)))
    h_a, h_b = pycbc.types.TimeSeries(h_a, delta_t=delta_t, epoch=geocent_time_a-len(h_a)*delta_t), pycbc.types.TimeSeries(h_b, delta_t=delta_t, epoch=geocent_time_a-len(h_a)*delta_t)

    # Adding the waveforms and generating the time series

    h = np.add(h_a,h_b)   
    h = pycbc.types.TimeSeries(h, delta_t=delta_t)

    return h


def psd_gen(det):
    """Reading the PSD files for the detectors"""

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

def mismatch(h, mchirp, eta, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time, det, psd):
    """Calculating the match/overlap defined as through a noise-weighted inner product in some frequency band maximized over a time and phase of coalescence"""

    mass_1, mass_2 = mconv(mchirp, eta)

    if math.isnan(mass_1) or math.isnan(mass_2) or math.isnan(spin1z) or math.isnan(spin2z) or mass_1 < 2. or mass_2 < 2 or mass_1/mass_2 < 1./18 or mass_1/mass_2 > 18 or mass_1+mass_2 > 800 or spin1z < -0.99 or spin1z > 0.99 or spin2z < -0.99 or spin2z > 0.99:
        log_mismatch = 1e6
    
    else:
        h_templ = wf(mchirp, eta, spin1z, spin2z, luminosity_distance, incl, phase, ra, dec, psi, geocent_time, det)
        
        # Resize the templates to the length of the waveform and interpolate the PSD
        
        h.resize(max(len(h), len(h_templ)))
        h_templ.resize(max(len(h), len(h_templ)))
        psd = pycbc.psd.interpolate(psd, 1/h.duration)

        # Evaluating the match through pycbc

        match = pycbc.filter.matchedfilter.match(h, h_templ, low_frequency_cutoff=minimum_frequency, psd=psd, high_frequency_cutoff=maximum_frequency, v1_norm=None, v2_norm=None)[0]
        log_mismatch = np.log10(1-match)   # Defining the log(mismatch)
    
    return log_mismatch 

def minimize_mismatch(fun_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0): 
   """Minimizing the Mismatch function through an adaptive Nelder-Mead optimization"""
   
   res = scipy.optimize.minimize(fun_mismatch, (mchirp_0, eta_0, spin1z_0, spin2z_0), method='Nelder-Mead', options={'adaptive':True}) 
   
   return res.fun

def pairs_match(mchirp_a, eta_a, spin1z_a, spin2z_a, luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a,
                mchirp_b, eta_b, spin1z_b, spin2z_b, luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b, 
                det, N_iter):
    
    """Initializing the Mismatch function over an iniitial array of truncated Gaussians around the injected parameters"""

    mass_1_a, mass_2_a = mconv(mchirp_a, eta_a)
    mass_1_b, mass_2_b = mconv(mchirp_b, eta_b)

    if math.isnan(mass_1_a) or math.isnan(mass_2_a) or math.isnan(spin1z_a) or math.isnan(spin2z_a) or mass_1_a < 2. or mass_2_a < 2 or mass_1_a/mass_2_a < 1./18 or mass_1_a/mass_2_a > 18 or mass_1_a+mass_2_a > 800 or spin1z_a < -0.99 or spin1z_a > 0.99 or spin2z_a < -0.99 or spin2z_a > 0.99:
        log_mismatch = 1e6

    elif math.isnan(mass_1_b) or math.isnan(mass_2_b) or math.isnan(spin1z_b) or math.isnan(spin2z_b) or mass_1_b < 2. or mass_2_b < 2 or mass_1_b/mass_2_b < 1./18 or mass_1_b/mass_2_b > 18 or mass_1_b+mass_2_b > 800 or spin1z_b < -0.99 or spin1z_b > 0.99 or spin2z_b < -0.99 or spin2z_b > 0.99:
        log_mismatch = 1e6

    else:
    
        h, psd = pairs_wf(mchirp_a, eta_a, spin1z_a, spin2z_a, luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a, mchirp_b, eta_b, spin1z_b, spin2z_b, luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b, det), psd_gen(det)

        # Initializing a function (returning log(mismatch)) for minimizing

        if waveform_metadata_a['H1']['optimal_SNR']>waveform_metadata_b['H1']['optimal_SNR']:
            fun_mismatch = lambda x: mismatch(h, x[0], x[1], x[2], x[3], luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a, det, psd)
        if waveform_metadata_b['H1']['optimal_SNR']>waveform_metadata_a['H1']['optimal_SNR']:
            fun_mismatch = lambda x: mismatch(h, x[0], x[1], x[2], x[3], luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b, det, psd)

        # Appending an array of initial points spread uniformily around the injected values

        mchirp_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(mchirp_a+mchirp_b)
        eta_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(eta_a+eta_b)
        spin1z_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(spin1z_a+spin1z_b)
        spin2z_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(spin2z_a+spin2z_b)

        idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<=0.25) & (spin1z_0>-0.99) & (spin1z_0<0.99) & (spin2z_0>-0.99) & (spin2z_0<0.99)

        mchirp_0 = np.append((mchirp_a+mchirp_b)/2, np.random.choice(mchirp_0[idx], int(N_iter)-1))
        eta_0 = np.append((eta_a+eta_b)/2, np.random.choice(eta_0[idx], int(N_iter)-1))
        spin1z_0 = np.append((spin1z_a+spin1z_b)/2, np.random.choice(spin1z_0[idx], int(N_iter)-1))
        spin2z_0 = np.append((spin2z_a+spin2z_b)/2, np.random.choice(spin2z_0[idx], int(N_iter)-1))
        
        # Minimizing the mismatch over all the initial set of points

        log_mismatch_arr = np.vectorize(minimize_mismatch)(fun_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0)
        match_arr = 1 - np.power(10, log_mismatch_arr)

        log_mismatch, match = np.min(log_mismatch_arr), 1 - 10**np.min(log_mismatch_arr)

        return log_mismatch, match, mchirp_0, eta_0, spin1z_0, spin2z_0, log_mismatch_arr, match_arr

# Importing the waveform metadata

waveform_metadata_a, waveform_metadata_b = pickle.load(open('git_overlap/src/output/overlap_injection/Waveform A Meta Data.pkl', 'rb')), pickle.load(open('git_overlap/src/output/overlap_injection/Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
for key,val in waveform_metadata_a['H1']['parameters'].items():
    exec(key +'_a' + '=val')
mchirp_a, eta_a = ((mass_1_a*mass_2_a)**(3/5))/((mass_1_a+mass_2_a)**(1/5)), (mass_1_a*mass_2_a)/((mass_1_a+mass_2_a)**2)
for key,val in waveform_metadata_b['H1']['parameters'].items():
    exec(key +'_b' + '=val')
mchirp_b, eta_b = ((mass_1_b*mass_2_b)**(3/5))/((mass_1_b+mass_2_b)**(1/5)), (mass_1_b*mass_2_b)/((mass_1_b+mass_2_b)**2)

# Evaluating the log(mismatch) values for the PAIRS

N_iter = int(5)   # Number of iterations of the initial points

log_mismatch_H1, match_H1, mchirp_arr_H1, eta_arr_H1, spin1z_arr_H1, spin2z_arr_H1, log_mismatch_arr_H1, match_arr_H1 = pairs_match(mchirp_a, eta_a, a_1_a, a_2_a, luminosity_distance_a, incl_a, phase_a, ra_a, dec_a, psi_a, geocent_time_a, mchirp_b, eta_b, a_1_b, a_2_b, luminosity_distance_b, incl_b, phase_b, ra_b, dec_b, psi_b, geocent_time_b, 'H1', N_iter)
min_idx = np.argmin(log_mismatch_arr_H1)
mchirpf_H1, etaf_H1, spin1zf_H1, spin2zf_H1 = mchirp_arr_H1[min_idx], eta_arr_H1[min_idx], spin1z_arr_H1[min_idx], spin2z_arr_H1[min_idx]

# Plotting the iterations with the recovered parameters over a cmap of match values
fig, ax = plt.subplots(1,2, figsize=(13, 6))
plt.suptitle('PAIRS; $\log$(Mismatch): %s, Match: %s'%(log_mismatch_H1, match_H1))

# Mchirp - Eta
mc12_c = ax[0].scatter(mchirp_arr_H1, eta_arr_H1, c=match_arr_H1, cmap='plasma')
ax[0].scatter((mchirp_a, mchirp_b), (eta_a, eta_b), facecolors = 'None', edgecolors='black')
ax[0].scatter(mchirpf_H1, etaf_H1, facecolors = 'None', edgecolors='black')
ax[0].text(mchirp_a, eta_a,'A', verticalalignment='top', horizontalalignment='left')
ax[0].text(mchirp_b, eta_b,'B', verticalalignment='top', horizontalalignment='left')
ax[0].text(mchirpf_H1, etaf_H1,'F', verticalalignment='bottom', horizontalalignment='right')
ax[0].set_xlabel('$\mathcal{M}$')
ax[0].set_ylabel('$\eta$')
ax[0].set_title('Mass Ratio $\eta$ $-$ Chirp Mass $\mathcal{M}$')
clb1 = plt.colorbar(mc12_c, ax=ax[0])
clb1.ax.set_title('$Match$')

# Spin1z -Spin2z
s12_c = ax[1].scatter(spin1z_arr_H1, spin2z_arr_H1, c=match_arr_H1, cmap='plasma')
ax[1].scatter((a_1_a, a_1_b), (a_2_a, a_2_b), facecolors = 'None', edgecolors='black')
ax[1].scatter(spin1zf_H1, spin2zf_H1, facecolors = 'None', edgecolors='black')
ax[1].text(a_1_a, a_2_a,'A', verticalalignment='top', horizontalalignment='left')
ax[1].text(a_1_b, a_2_b,'B', verticalalignment='top', horizontalalignment='left')
ax[1].text(spin1zf_H1, spin2zf_H1,'F', verticalalignment='bottom', horizontalalignment='right')
ax[1].set_xlabel('$a^z_1$')
ax[1].set_ylabel('$a^z_2$')
ax[1].set_title('Spins: $a_1^z-a_2^z$')
clb2 = plt.colorbar(s12_c, ax=ax[1])
clb2.ax.set_title('$Match$')

fig.tight_layout()
plt.savefig('git_overlap/src/output/overlap_nm/NM: PAIRS.png')
plt.close()

mass_1f_H1, mass_2f_H1 = mconv(mchirpf_H1, etaf_H1)

# Printing the required results
print('Injected - SINGLES A: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_a, 3), np.round(eta_a, 3), np.round(a_1_a, 3), np.round(a_2_a, 3), np.round(mass_1_a, 3), np.round(mass_2_a, 3)))
print('Injected - SINGLES B: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s, Δt: %s'%(np.round(mchirp_b, 3), np.round(eta_b, 3), np.round(a_1_b, 3), np.round(a_2_b, 3), np.round(mass_1_b, 3), np.round(mass_2_b, 3), np.round(delta_b, 3)))
print('Recovered - PAIRS: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirpf_H1, 3), np.round(etaf_H1, 3), np.round(spin1zf_H1, 3), np.round(spin2zf_H1, 3), np.round(mass_1f_H1, 3), np.round(mass_2f_H1, 3)))