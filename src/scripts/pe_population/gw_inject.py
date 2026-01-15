#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# Generate Overlapping BBH Injections from a population and inject into detectors (H1, L1, V1)

import numpy as np
from tqdm import tqdm
import itertools
from pesummary.io import read
from pycbc.frame import write_frame
from pycbc import filter
import pycbc.psd
import pycbc.types
import os
import sys 

sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import waveforms as wf
import FF_computation as ff
from FF_computation import mchirp_q_to_m1m2

import matplotlib.pyplot as plt

# Initialize Waveform Generator
gen = wf.PairsWaveformGeneration()
plt.style.use('default')

# Configuration
f_lower = 20.0
f_ref = 50.0
duration = 32.0
sampling_frequency = 4096.0
delta_t = 1.0 / sampling_frequency
f_higher = sampling_frequency / 2.0

kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, f_high=f_higher, delta_t=delta_t)
ff_config = dict(f_low=f_lower, f_high=f_higher, psd=None, n_iters=15) # Increased iters slightly for better convergence

# Injection Grid
mchirpratio, snrratio = [.5, 1., 2.], [.5, 1.]
delta_tc = [-.1, -.05, -.03, -.02, -.01, .01, .02, .03, .05, .1]
combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))

# Load Posterior Data
injection_a = read('/home/nishkal.rao/git_overlap/src/data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']
injection_b = read('/home/nishkal.rao/git_overlap/src/data/IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']

nmap_a = np.argmax(injection_a['log_likelihood']+injection_a['log_prior'])
nmap_b = np.argmax(injection_b['log_likelihood']+injection_b['log_prior'])

# Setup Base Parameters
waveform_params_a = {
    'approximant': 'IMRPhenomXPHM',
    'f_lower': f_lower,
    'f_ref': f_ref,
    'delta_t': delta_t
}
for key in injection_a.keys():
    waveform_params_a[key] = injection_a[key][nmap_a]

waveform_params_b = {
    'approximant': 'IMRPhenomXPHM',
    'f_lower': f_lower,
    'f_ref': f_ref,
    'delta_t': delta_t
}
for key in injection_b.keys():
    waveform_params_b[key] = injection_b[key][nmap_b]

# Batch Processing Logic
N, queue = len(combinations), 60
try:
    k = int(sys.argv[1])
except IndexError:
    k = 0
combinations = combinations[k*int(N/queue):(k+1)*int(N/queue)]

# Create Output Directories if they don't exist
os.makedirs('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/', exist_ok=True)
os.makedirs('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/', exist_ok=True)

# Main Loop
for i in tqdm(range(len(combinations))):
    
    # 1. Setup Injection C (Scaled B)
    waveform_params_c = waveform_params_b.copy()
    waveform_params_c['mass_1'] = waveform_params_a['mass_1']*combinations[i][0]
    waveform_params_c['mass_2'] = waveform_params_a['mass_2']*combinations[i][0]
    waveform_params_c['geocent_time'] = waveform_params_a['geocent_time'] + combinations[i][2]

    # 2. SNR Scaling
    # Generate temporary waveforms to calculate SNR
    ht_a = gen.wf_td(waveform_params_a, **kwargs)
    ht_b = gen.wf_td(waveform_params_c, **kwargs)
    
    snr_a, snr_b = np.zeros(3), np.zeros(3)
    
    # Calculate initial SNR
    for idx, det in enumerate(['H1', 'L1', 'V1']):
        psd = pycbc.psd.aLIGOZeroDetHighPower(len(ht_a[det]), ht_a[det].delta_f, f_lower)   
        snr_at = filter.matchedfilter.matched_filter(ht_a[det], ht_a[det], psd=psd, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        
        psd_b = pycbc.psd.aLIGOZeroDetHighPower(len(ht_b[det]), ht_b[det].delta_f, f_lower)   
        snr_bt = filter.matchedfilter.matched_filter(ht_b[det], ht_b[det], psd=psd_b, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        
        snr_a[idx] = abs(snr_at[snr_at.numpy().argmax()])
        snr_b[idx] = abs(snr_bt[snr_bt.numpy().argmax()])
    
    # Scale Distance to target SNR = 30 (ref)
    waveform_params_a['snr_det'] = np.sqrt(np.sum(snr_a**2))
    waveform_params_c['snr_det'] = np.sqrt(np.sum(snr_b**2))
    
    waveform_params_a['luminosity_distance'] = waveform_params_a['luminosity_distance'] * waveform_params_a['snr_det'] / 30.0  
    waveform_params_c['luminosity_distance'] = waveform_params_c['luminosity_distance'] * waveform_params_c['snr_det'] / (30.0 * combinations[i][1])    

    # Regenerate Scaled Waveforms
    ht_a = gen.wf_td(waveform_params_a, **kwargs)
    ht_b = gen.wf_td(waveform_params_c, **kwargs)
    
    # Save Injection Params
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/SINGLES_A_{}_{}_{}.npy'.format(combinations[i][0], combinations[i][1], combinations[i][2]), waveform_params_a)
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/SINGLES_B_{}_{}_{}.npy'.format(combinations[i][0], combinations[i][1], combinations[i][2]), waveform_params_c)

    # 3. Generate Pairs and Save Frames
    ht_pairs = gen.pairs_td(waveform_params_a, waveform_params_c, **kwargs)
    
    for idx, det in enumerate(['H1', 'L1', 'V1']):
        outfile = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/PAIRS_{}_{}_{}_{}.gwf'.format(det, combinations[i][0], combinations[i][1], combinations[i][2])
        write_frame(outfile, "{}:PyCBC_Injection".format(det), ht_pairs[det])

        # Plot Injection
        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_pairs[det].sample_times, ht_pairs[det], 'm', label='Pairs $h_A + h_B$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        ax.set_xlim(waveform_params_a['geocent_time'] - 0.5, waveform_params_c['geocent_time'] + 0.3)
        plt.title(r'Injection ({})'.format(det))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/Injection_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()

        # ==========================================================================================
        # 4. Fitting Factors
        # ==========================================================================================

        # A. 2D Recovery (Singles v Pairs)
        # ------------------------------------------------------------------------------------------
        FF_res = ff.compute_fitting_factor(ht_pairs[det], wf_model='2D', apx="IMRPhenomXPHM", **ff_config)
        
        # Save Results: [FF, m_chirp, q]
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_2D_{}_{}_{}_{}.npy'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]), [FF_res[0], FF_res[1][0], FF_res[1][1]])
        
        # Recovered Waveform Generation
        recovered_params = waveform_params_a.copy()
        recovered_params['mass_1'], recovered_params['mass_2'] = mchirp_q_to_m1m2(FF_res[1][0], FF_res[1][1])
        ht_recovered = gen.wf_td(injection_parameters=recovered_params, **kwargs)
        
        # Align Peak
        shift = ht_pairs[det].sample_times[np.argmax(np.abs(ht_pairs[det]))] - ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]
        ht_recovered[det].start_time += shift

        # Plot 2D Recovery
        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_pairs[det].sample_times, ht_pairs[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered[det].sample_times, ht_recovered[det], 'k', label='Recovery $\\tilde{h}_{0}$ (2D)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        rec_time = ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]        
        ax.set_xlim(rec_time - 0.5, rec_time + 0.3)
        plt.title(r'2D REC ({}): Match: ${:.3f}$ | $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(det, FF_res[0], FF_res[1][0], FF_res[1][1]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_2D_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()

        # B. Microlensed 4D Recovery (ML_4D)
        # ------------------------------------------------------------------------------------------
        # We perform FF on the TD pairs using the ML_4D model. 
        # Note: wf_ml_fd returns frequency series, FF computation handles this.
        
        FF_res_ml = ff.compute_fitting_factor(ht_pairs[det], wf_model='ML_4D', apx="IMRPhenomXPHM", **ff_config)
        
        # Save Results: [FF, m_chirp, q, Mlz, y]
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_ML_4D_{}_{}_{}_{}.npy'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]), 
                [FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3]])

        recovered_params_ml = waveform_params_a.copy()
        recovered_params_ml['mass_1'], recovered_params_ml['mass_2'] = mchirp_q_to_m1m2(FF_res_ml[1][0], FF_res_ml[1][1])

        # Generate ML FD and convert to TD for plotting
        hf_recovered_ml = gen.wf_ml_fd(injection_parameters=recovered_params_ml, Ml_z=FF_res_ml[1][2], y=FF_res_ml[1][3], **kwargs)
        ht_recovered_ml = hf_recovered_ml[det].to_timeseries()
        
        # Align
        shift_ml = ht_pairs[det].sample_times[np.argmax(np.abs(ht_pairs[det]))] - ht_recovered_ml.sample_times[np.argmax(np.abs(ht_recovered_ml))]
        ht_recovered_ml.start_time += shift_ml

        # Plot ML Recovery
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
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_ML_4D_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()

        # C. Eccentric 4D Recovery (EC_4D)
        # ------------------------------------------------------------------------------------------
        # Replaces EC_0 and EC_3D. The optimizer seeks best e.
        
        FF_res_ecc = ff.compute_fitting_factor(ht_pairs[det], wf_model='EC_4D', apx="IMRPhenomXPHM", **ff_config)
        
        # Save Results: [FF, m_chirp, q, ecc, anomaly]
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_EC_4D_{}_{}_{}_{}.npy'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]), 
                [FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2], FF_res_ecc[1][3]])

        recovered_params_ecc = waveform_params_a.copy()
        recovered_params_ecc['mass_1'], recovered_params_ecc['mass_2'] = mchirp_q_to_m1m2(FF_res_ecc[1][0], FF_res_ecc[1][1])

        ht_recovered_ecc_dict = gen.wf_ecc_td(injection_parameters=recovered_params_ecc, e=FF_res_ecc[1][2], anomaly=FF_res_ecc[1][3], **kwargs)
        ht_recovered_ecc = ht_recovered_ecc_dict[det]

        # Align
        shift_ecc = ht_pairs[det].sample_times[np.argmax(np.abs(ht_pairs[det]))] - ht_recovered_ecc.sample_times[np.argmax(np.abs(ht_recovered_ecc))]
        ht_recovered_ecc.start_time += shift_ecc

        # Plot ECC Recovery
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
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_EC_4D_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()