#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# Generate Overlapping BBH Injections from a population and inject into detectors (H1, L1, V1)

import numpy as np
from tqdm import tqdm
import deepdish
from pycbc.frame import write_frame
from pycbc import filter
import pycbc.psd
import os
import sys 

sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import waveforms as wf
import FF_computation as ff
from FF_computation import mchirp_q_to_m1m2, m1m2_to_mchirp

import matplotlib.pyplot as plt

# Initialize Generator
gen = wf.PairsWaveformGeneration()
plt.style.use('default')

# Configuration
f_lower = 20.0
f_ref = 50.0
sampling_frequency = 4096.0
delta_t = 1.0 / sampling_frequency
f_higher = sampling_frequency / 2.0

kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, f_high=f_higher, delta_t=delta_t)
ff_config = dict(f_low=f_lower, f_high=f_higher, psd=None, n_iters=15)

# Load Population Injections
injection = dict(deepdish.io.load('/home/nishkal.rao/git_overlap/src/output/injections/injections.hdf5')['injections'])
# Create variables in local scope just in case, though access via dict is safer
for key, val in injection.items():
    exec(key + '=val')
injection['chirp_mass'] = m1m2_to_mchirp(injection['mass_1'], injection['mass_2'])

try:
    k = int(sys.argv[1])
except IndexError:
    k = 0
N, queue = 5000, 5000
indices = np.arange(k*int(N/queue), (k+1)*int(N/queue))

# Create Dirs
os.makedirs('/home/nishkal.rao/git_overlap/src/output/pe_population/population_injections/', exist_ok=True)
os.makedirs('/home/nishkal.rao/git_overlap/src/output/pe_population/population/', exist_ok=True)

for idx, i in tqdm(enumerate(indices)):

    # 1. Setup Parameters A and B
    waveform_params_a, waveform_params_b = {}, {}
    for key, val in injection.items():
        waveform_params_a[key] = val[i]
        waveform_params_b[key] = val[N-1-i]

    # 2. SNR Pre-computation for Distance Scaling
    ht_a = gen.wf_td(waveform_params_a, **kwargs)
    ht_b = gen.wf_td(waveform_params_b, **kwargs)
    
    snr_a, snr_b = np.zeros(3), np.zeros(3)
    # Estimate SNR using sigma for initial scaling (faster than match filtering)
    for ids, det in enumerate(['H1', 'L1', 'V1']):
        snr_a[ids] = filter.matchedfilter.sigma(ht_a[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        snr_b[ids] = filter.matchedfilter.sigma(ht_b[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
    
    # Scale A Distance
    waveform_params_a['luminosity_distance'] = waveform_params_a['luminosity_distance'] * np.sqrt(np.sum(snr_a**2))/24.0

    # Modify B Parameters (Overlap Configuration)
    waveform_params_b['chirp_mass'] = waveform_params_a['chirp_mass'] * np.random.uniform(0.5, 2)
    # Mass ratio stays from population
    waveform_params_b['luminosity_distance'] = waveform_params_a['luminosity_distance'] * (np.sum(snr_a**2)/np.sum(snr_b**2))**(0.25) # Approx scaling
    waveform_params_b['geocent_time'] = waveform_params_a['geocent_time'] + np.random.uniform(-0.1, 0.1)
    
    # Recalculate Precise SNR for Final Scaling
    ht_a = gen.wf_td(waveform_params_a, **kwargs)
    ht_b = gen.wf_td(waveform_params_b, **kwargs)
    
    snr_a, snr_b = np.zeros(3), np.zeros(3)
    for ids, det in enumerate(['H1', 'L1', 'V1']):
        psd = pycbc.psd.aLIGOZeroDetHighPower(len(ht_a[det]), ht_a[det].delta_f, f_lower)   
        # Auto-match to get peak SNR
        snr_at = filter.matchedfilter.matched_filter(ht_a[det], ht_a[det], psd=psd, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        
        psd_b = pycbc.psd.aLIGOZeroDetHighPower(len(ht_b[det]), ht_b[det].delta_f, f_lower)   
        snr_bt = filter.matchedfilter.matched_filter(ht_b[det], ht_b[det], psd=psd_b, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        
        snr_a[ids] = abs(snr_at[snr_at.numpy().argmax()])
        snr_b[ids] = abs(snr_bt[snr_bt.numpy().argmax()])
        
    waveform_params_a['snr_det'], waveform_params_b['snr_det'] = np.sqrt(np.sum(snr_a**2)), np.sqrt(np.sum(snr_b**2))

    # Final Distance Scaling to SNR=30 Ref
    waveform_params_a['luminosity_distance'] = waveform_params_a['luminosity_distance'] * waveform_params_a['snr_det'] / 30.0  
    # Scale B relative to A's SNR ratio
    # If we want B to have a specific SNR ratio relative to A, we adjust here. 
    # Current logic seems to target SNR=30 for A, and scales B based on the intrinsic loudness difference.
    scale_factor_b = waveform_params_b['snr_det'] / (30.0 * np.sqrt(np.sum(snr_b**2)/np.sum(snr_a**2)))
    waveform_params_b['luminosity_distance'] = waveform_params_b['luminosity_distance'] * scale_factor_b

    # Final Waveform Generation
    ht_a = gen.wf_td(waveform_params_a, **kwargs)
    ht_b = gen.wf_td(waveform_params_b, **kwargs)
    
    # Save Parameters
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/population_injections/SINGLES_A_{}.npy'.format(i), waveform_params_a)
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/population_injections/SINGLES_B_{}.npy'.format(i), waveform_params_b)

    # 3. Process Detectors
    for ids, det in enumerate(['H1', 'L1', 'V1']):
        ht_pairs = gen.pairs_td(waveform_params_a, waveform_params_b, **kwargs)

        # Plot Injection
        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_pairs[det].sample_times, ht_pairs[det], 'm', label='Pairs $h_A + h_B$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        ax.set_xlim(waveform_params_a['geocent_time'] - 0.5, waveform_params_b['geocent_time'] + 0.3)
        plt.title(r'Injection ({})'.format(det))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/population/Injection_{}_{}.png'.format(det, i))
        plt.close()

        # ==========================================================================================
        # 4. Fitting Factors
        # ==========================================================================================

        # A. 2D Recovery
        # ------------------------------------------------------------------------------------------
        FF_res = ff.compute_fitting_factor(ht_pairs[det], wf_model='2D', apx="IMRPhenomXPHM", **ff_config)
        
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/population/FF_2D_{}_{}.npy'.format(det, i), [FF_res[0], FF_res[1][0], FF_res[1][1]])
        
        recovered_params = waveform_params_a.copy()
        recovered_params['mass_1'], recovered_params['mass_2'] = mchirp_q_to_m1m2(FF_res[1][0], FF_res[1][1])

        ht_recovered = gen.wf_td(injection_parameters=recovered_params, **kwargs)
        
        shift = ht_pairs[det].sample_times[np.argmax(np.abs(ht_pairs[det]))] - ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]
        ht_recovered[det].start_time += shift

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_pairs[det].sample_times, ht_pairs[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered[det].sample_times, ht_recovered[det], 'k', label='Recovery $\\tilde{h}_0$ (2D)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        rec_time = ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]
        ax.set_xlim(rec_time - 0.5, rec_time + 0.3)
        plt.title(r'2D REC ({}): Match: ${:.3f}$ | $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(det, FF_res[0], FF_res[1][0], FF_res[1][1]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/population/FF_2D_{}_{}.png'.format(det, i))
        plt.close()

        # B. Microlensed 4D Recovery
        # ------------------------------------------------------------------------------------------
        FF_res_ml = ff.compute_fitting_factor(ht_pairs[det], wf_model='ML_4D', apx="IMRPhenomXPHM", **ff_config)
        
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/population/FF_ML_4D_{}_{}.npy'.format(det, i), [FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3]])

        recovered_params_ml = waveform_params_a.copy()
        recovered_params_ml['mass_1'], recovered_params_ml['mass_2'] = mchirp_q_to_m1m2(FF_res_ml[1][0], FF_res_ml[1][1])

        hf_recovered_ml = gen.wf_ml_fd(injection_parameters=recovered_params_ml, Ml_z=FF_res_ml[1][2], y=FF_res_ml[1][3], **kwargs)
        ht_recovered_ml = hf_recovered_ml[det].to_timeseries()
        
        shift_ml = ht_pairs[det].sample_times[np.argmax(np.abs(ht_pairs[det]))] - ht_recovered_ml.sample_times[np.argmax(np.abs(ht_recovered_ml))]
        ht_recovered_ml.start_time += shift_ml

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_pairs[det].sample_times, ht_pairs[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_ml.sample_times, ht_recovered_ml, 'k', label='Recovery $\\tilde{h}_{ML}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        rec_time_ml = ht_recovered_ml.sample_times[np.argmax(np.abs(ht_recovered_ml))]
        ax.set_xlim(rec_time_ml - 0.5, rec_time_ml + 0.3)
        plt.title(r'ML_4D REC ({}): Match: ${:.3f}$ | $\mathcal{{M}}={:.3f}, q={:.3f}, M_{{lz}}={:.1f}, y={:.3f}$'.format(
            det, FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/population/FF_ML_4D_{}_{}.png'.format(det, i))
        plt.close()

        # C. Eccentric 4D Recovery
        # ------------------------------------------------------------------------------------------
        FF_res_ecc = ff.compute_fitting_factor(ht_pairs[det], wf_model='EC_4D', apx="IMRPhenomXPHM", **ff_config)
        
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/population/FF_EC_4D_{}_{}.npy'.format(det, i), [FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2], FF_res_ecc[1][3]])

        recovered_params_ecc = waveform_params_a.copy()
        recovered_params_ecc['mass_1'], recovered_params_ecc['mass_2'] = mchirp_q_to_m1m2(FF_res_ecc[1][0], FF_res_ecc[1][1])

        ht_recovered_ecc_dict = gen.wf_ecc_td(injection_parameters=recovered_params_ecc, e=FF_res_ecc[1][2], anomaly=FF_res_ecc[1][3], **kwargs)
        ht_recovered_ecc = ht_recovered_ecc_dict[det]
        
        shift_ecc = ht_pairs[det].sample_times[np.argmax(np.abs(ht_pairs[det]))] - ht_recovered_ecc.sample_times[np.argmax(np.abs(ht_recovered_ecc))]
        ht_recovered_ecc.start_time += shift_ecc

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_pairs[det].sample_times, ht_pairs[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_ecc.sample_times, ht_recovered_ecc, 'k', label='Recovery $\\tilde{h}_{EC}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        rec_time_ecc = ht_recovered_ecc.sample_times[np.argmax(np.abs(ht_recovered_ecc))]
        ax.set_xlim(rec_time_ecc - 0.5, rec_time_ecc + 0.3)
        plt.title(r'EC_4D REC ({}): Match: ${:.3f}$ | $\mathcal{{M}}={:.3f}, q={:.3f}, e={:.3f}$'.format(
            det, FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/population/FF_EC_4D_{}_{}.png'.format(det, i))
        plt.close()