#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# Generate Overlapping BBH Injections from a population and inject into detectors (H1, L1, V1)

import numpy as np
from tqdm import tqdm
import itertools
import deepdish
from pycbc.frame import write_frame
from pycbc import filter

import sys 
sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import waveforms as wf
import FF_computation as ff
from FF_computation import mchirp_q_to_m1m2, m1m2_to_mchirp

import matplotlib.pyplot as plt
# plt.rcParams.update({"text.usetex": True,
#     "font.family": "sans-serif",
#     "axes.formatter.use_mathtext": True,
#     "axes.formatter.limits": (-3, 3)
# })

gen = wf.PairsWaveformGeneration()
plt.style.use('default')

f_lower = 20.0
f_ref = 50.0
sampling_frequency = 4096.0
delta_t = 1.0 / sampling_frequency
f_higher = sampling_frequency / 2.0

kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, delta_t=delta_t)

injection = dict(deepdish.io.load('/home/nishkal.rao/git_overlap/src/output/injections/injections.hdf5')['injections'])
for key, val in injection.items():
    exec(key + '=val')
mchirp = m1m2_to_mchirp(mass_1, mass_2)
injection['chirp_mass'] = mchirp

k = int(sys.argv[1])
N, queue = 5000, 5000
indices = np.arange(k*int(N/queue), (k+1)*int(N/queue))

injections = {}
for idx, i in tqdm(enumerate(indices)):

    waveform_params_a, waveform_params_b = {}, {}
    for key, val in injection.items():
        waveform_params_a[key] = val[i]
        waveform_params_b[key] = val[N-1-i]

    ht_a, ht_b = gen.wf_td(waveform_params_a, **kwargs), gen.wf_td(waveform_params_b, **kwargs)
    
    snr_a, snr_b = np.zeros(3), np.zeros(3)
    for ids, det in enumerate(['H1', 'L1', 'V1']):
        snr_a[ids] = filter.matchedfilter.sigma(ht_a[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        snr_b[ids] = filter.matchedfilter.sigma(ht_b[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
    waveform_params_a['luminosity_distance'] = waveform_params_a['luminosity_distance'] * np.sqrt(np.sum(snr_a**2))/24

    waveform_params_b['chirp_mass'] = waveform_params_a['chirp_mass'] * np.random.uniform(0.5, 2)
    waveform_params_b['mass_ratio'] = waveform_params_a['mass_ratio']
    waveform_params_b['luminosity_distance'] = waveform_params_a['luminosity_distance'] * (np.sum(snr_a**2)/np.sum(snr_b**2))**1/4
    waveform_params_b['geocent_time'] = waveform_params_a['geocent_time'] + np.random.uniform(-0.1, 0.1)
    
    ht_a, ht_b = gen.wf_td(waveform_params_a, **kwargs), gen.wf_td(waveform_params_b, **kwargs)

    snr_a, snr_b = np.zeros(3), np.zeros(3)
    for ids, det in enumerate(['H1', 'L1', 'V1']):
        snr_a[ids] = filter.matchedfilter.sigma(ht_a[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        snr_b[ids] = filter.matchedfilter.sigma(ht_b[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)    
    waveform_params_a['snr_det'], waveform_params_b['snr_det'] = np.sqrt(np.sum(snr_a**2)), np.sqrt(np.sum(snr_b**2))  
    
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population_injections/SINGLES_A_{}.npy'.format(i), waveform_params_a)
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population_injections/SINGLES_B_{}.npy'.format(i), waveform_params_b)

    for ids, det in enumerate(['H1', 'L1', 'V1']):
        ht = gen.pairs_td(waveform_params_a, waveform_params_b, **kwargs)
        FF_res, param_history = ff.compute_fitting_factor(ht[det], wf_model='2D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_2D_{}_{}.npy'.format(det, i), [FF_res[0], FF_res[1][0], FF_res[1][1]])
        
        recovered_params = waveform_params_a.copy()
        recovered_params['mass_1'], recovered_params['mass_2'] = mchirp_q_to_m1m2(FF_res[1][0], FF_res[1][1])

        ht_recovered = gen.wf_td(injection_parameters = recovered_params, **kwargs)
        ht_recovered[det].start_time += ht[det].sample_times[np.argmax(np.abs(ht[det]))] - ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht[det].sample_times, ht[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered[det].sample_times, ht_recovered[det], 'k', label='Recovery $\\tilde{h}_0$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params['geocent_time'] = ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]        
        ax.set_xlim(recovered_params['geocent_time'] - 0.5, recovered_params['geocent_time'] + 0.3)
        plt.title(r'REC ({}): Recovered Waveform with Match: ${:.3f}$ and 2D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(det, FF_res[0], FF_res[1][0], FF_res[1][1]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_2D_{}_{}.png'.format(det, i))
        plt.close()

        param_history = np.array(param_history)
        chirp_mass_vals, q_vals, match_vals = param_history[:, 0], param_history[:, 1], param_history[:, 2]

        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sc1 = ax[0].scatter(range(len(chirp_mass_vals)), chirp_mass_vals, c=match_vals, cmap='plasma', label='Chirp Mass $\\mathcal{M}$', zorder=3)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('$\\mathcal{M}$')
        ax[0].axhline(waveform_params_a['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_A$', zorder=1)
        ax[0].axhline(waveform_params_b['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_B$', zorder=1)
        ax[0].axhline(FF_res[1][0], color='r', linestyle='-', label='Final $\\mathcal{M}$', zorder=2)
        ax[0].set_title('Chirp Mass $\\mathcal{M}$')
        sc2 = ax[1].scatter(range(len(q_vals)), q_vals, c=match_vals, cmap='plasma', label='Mass Ratio $q$', zorder=3)
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('$q$')
        ax[1].axhline(waveform_params_a['mass_ratio'], color='k', linestyle='--', label='$q_A$', zorder=1)
        ax[1].axhline(waveform_params_b['mass_ratio'], color='k', linestyle='--', label='$q_B$', zorder=1)
        ax[1].axhline(FF_res[1][1], color='r', linestyle='-', label='Final $q$', zorder=2)
        ax[1].set_title('Mass Ratio $q$')
        cbar = fig.colorbar(sc2, ax=ax[1], orientation='vertical', label='Match')
        cbar.set_label('Match')
        plt.suptitle('Convergence of Chirp Mass, Mass Ratio for {} sample {}'.format(det, i))
        fig.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_2D_convergence_{}_{}.png'.format(det, i))
        plt.close()

        ####################################################################################################################################################################        

        hf_a_sl, hf_b_sl = gen.wf_fd(injection_parameters=waveform_params_a, imagetype=1, **kwargs), gen.wf_sl_fd(injection_parameters=waveform_params_b, imagetype=1, **kwargs)
        hf_sl = gen.pairs_fd(waveform_params_a, waveform_params_b, **kwargs)
        FF_res_sl, param_history_sl = ff.compute_fitting_factor(hf_sl[det], wf_model='SL_2D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_SL_2D_{}_{}.npy'.format(det, i), [FF_res_sl[0], FF_res_sl[1][0], FF_res_sl[1][1]])

        recovered_params_sl = waveform_params_a.copy()
        recovered_params_sl['mass_1'], recovered_params_sl['mass_2'] = mchirp_q_to_m1m2(FF_res_sl[1][0], FF_res_sl[1][1])

        hf_recovered_sl = gen.wf_sl_fd(injection_parameters = recovered_params_sl, imagetype = 2, **kwargs)
        ht_recovered_sl, ht_a_sl, ht_b_sl, ht_sl = {}, {}, {}, {}
        ht_recovered_sl[det], ht_sl[det] = hf_recovered_sl[det].to_timeseries(), hf_sl[det].to_timeseries()
        ht_a_sl[det], ht_b_sl[det] = hf_a_sl[det].to_timeseries(), hf_b_sl[det].to_timeseries()

        ht_sl[det].start_time, ht_a_sl[det].start_time, ht_b_sl[det].start_time = ht[det].start_time, ht_a[det].start_time, ht_b[det].start_time
        ht_recovered_sl[det].start_time += ht_sl[det].sample_times[np.argmax(np.abs(ht_sl[det]))] - ht_recovered_sl[det].sample_times[np.argmax(np.abs(ht_recovered_sl[det]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a_sl[det].sample_times, ht_a_sl[det], 'b--', label='Singles $h_A$')
        ax.plot(ht_b_sl[det].sample_times, ht_b_sl[det], 'r--', label='Singles $h_B$')
        ax.plot(ht_b_sl[det].sample_times, ht_b_sl[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_sl[det].sample_times, ht_recovered_sl[det], 'k', label='Recovery $\\tilde{h}_{II}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_sl['geocent_time'] = ht_recovered_sl[det].sample_times[np.argmax(np.abs(ht_recovered_sl[det]))]
        ax.set_xlim(recovered_params_sl['geocent_time'] - 0.5, recovered_params_sl['geocent_time'] + 0.3)
        plt.title(r'SL ({}): Recovered Waveform with Match: ${:.3f}$ and 2D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(det, FF_res_sl[0], FF_res_sl[1][0], FF_res_sl[1][1]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_SL_2D_{}_{}.png'.format(det, i))
        plt.close()

        param_history_sl = np.array(param_history_sl)
        chirp_mass_vals, q_vals, match_vals = param_history_sl[:, 0], param_history_sl[:, 1], param_history_sl[:, 2]

        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sc1 = ax[0].scatter(range(len(chirp_mass_vals)), chirp_mass_vals, c=match_vals, cmap='plasma', label='Chirp Mass $\\mathcal{M}$', zorder=3)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('$\\mathcal{M}$')
        ax[0].axhline(waveform_params_a['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_A$', zorder=1)
        ax[0].axhline(waveform_params_b['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_B$', zorder=1)
        ax[0].axhline(FF_res_sl[1][0], color='r', linestyle='-', label='Final $\\mathcal{M}$', zorder=2)
        ax[0].set_title('Chirp Mass $\\mathcal{M}$')
        sc2 = ax[1].scatter(range(len(q_vals)), q_vals, c=match_vals, cmap='plasma', label='Mass Ratio $q$', zorder=3)
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('$q$')
        ax[1].axhline(waveform_params_a['mass_ratio'], color='k', linestyle='--', label='$q_A$', zorder=1)
        ax[1].axhline(waveform_params_b['mass_ratio'], color='k', linestyle='--', label='$q_B$', zorder=1)
        ax[1].axhline(FF_res_sl[1][1], color='r', linestyle='-', label='Final $q$', zorder=2)
        ax[1].set_title('Mass Ratio $q$')
        cbar = fig.colorbar(sc2, ax=ax[1], orientation='vertical', label='Match')
        cbar.set_label('Match')
        plt.suptitle('Convergence of Chirp Mass, Mass Ratio for {}  sample {}'.format(det, i))
        fig.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_SL_convergence_{}_{}.png'.format(det, i))
        plt.close()

        ####################################################################################################################################################################

        hf_a_slc, hf_b_slc = gen.wf_fd(injection_parameters=waveform_params_a, imagetype=1, **kwargs), gen.wf_sl_fd(injection_parameters=waveform_params_b, imagetype=1, **kwargs)
        hf_slc = gen.pairs_fd(waveform_params_a, waveform_params_b, **kwargs)
        FF_res_slc, param_history_slc = ff.compute_fitting_factor(hf_slc[det], wf_model='SLC_3D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_SLC_3D_{}_{}.npy'.format(det, i), [FF_res_slc[0], FF_res_slc[1][0], FF_res_slc[1][1], FF_res_slc[1][2]])

        recovered_params_slc = waveform_params_a.copy()
        recovered_params_slc['mass_1'], recovered_params_slc['mass_2'] = mchirp_q_to_m1m2(FF_res_slc[1][0], FF_res_slc[1][1])

        hf_recovered_slc = gen.wf_sl_fd(injection_parameters = recovered_params_slc, imagetype=FF_res_slc[1][2], **kwargs)
        ht_recovered_slc, ht_a_slc, ht_b_slc, ht_slc = {}, {}, {}, {}
        ht_recovered_slc[det], ht_slc[det] = hf_recovered_slc[det].to_timeseries(), hf_slc[det].to_timeseries()
        ht_a_slc[det], ht_b_slc[det] = hf_a_slc[det].to_timeseries(), hf_b_slc[det].to_timeseries()

        ht_slc[det].start_time, ht_a_slc[det].start_time, ht_b_slc[det].start_time = ht[det].start_time, ht_a[det].start_time, ht_b[det].start_time
        ht_recovered_slc[det].start_time += ht_slc[det].sample_times[np.argmax(np.abs(ht_slc[det]))] - ht_recovered_slc[det].sample_times[np.argmax(np.abs(ht_recovered_slc[det]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a_slc[det].sample_times, ht_a_slc[det], 'b--', label='Singles $h_A$')
        ax.plot(ht_b_slc[det].sample_times, ht_b_slc[det], 'r--', label='Singles $h_B$')
        ax.plot(ht_b_slc[det].sample_times, ht_b_slc[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_slc[det].sample_times, ht_recovered_slc[det], 'k', label='Recovery $\\tilde{h}_{III}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_slc['geocent_time'] = ht_recovered_slc[det].sample_times[np.argmax(np.abs(ht_recovered_slc[det]))]
        ax.set_xlim(recovered_params_slc['geocent_time'] - 0.5, recovered_params_slc['geocent_time'] + 0.3)
        plt.title(r'SLC ({}): Recovered Waveform with Match: ${:.3f}$ and 3D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $n_j = {}\pi/2$'.format(det, FF_res_slc[0], FF_res_slc[1][0], FF_res_slc[1][1], FF_res_slc[1][2]-1))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_SLC_3D_{}_{}.png'.format(det, i))
        plt.close()

        param_history_slc = np.array(param_history_slc)
        chirp_mass_vals, q_vals, match_vals, imagetype_vals = param_history_slc[:, 0], param_history_slc[:, 1], param_history_slc[:, 2], param_history_slc[:, 3]

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        sc1 = ax[0].scatter(range(len(chirp_mass_vals)), chirp_mass_vals, c=match_vals, cmap='plasma', label='Chirp Mass $\\mathcal{M}$', zorder=3)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('$\\mathcal{M}$')
        ax[0].axhline(waveform_params_a['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_A$', zorder=1)
        ax[0].axhline(waveform_params_b['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_B$', zorder=1)
        ax[0].axhline(FF_res_slc[1][0], color='r', linestyle='-', label='Final $\\mathcal{M}$', zorder=2)
        ax[0].set_title('Chirp Mass $\\mathcal{M}$')
        sc2 = ax[1].scatter(range(len(q_vals)), q_vals, c=match_vals, cmap='plasma', label='Mass Ratio $q$', zorder=3)
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('$q$')
        ax[1].axhline(waveform_params_a['mass_ratio'], color='k', linestyle='--', label='$q_A$', zorder=1)
        ax[1].axhline(waveform_params_b['mass_ratio'], color='k', linestyle='--', label='$q_B$', zorder=1)
        ax[1].axhline(FF_res_slc[1][1], color='r', linestyle='-', label='Final $q$', zorder=2)
        ax[1].set_title('Mass Ratio $q$')
        sc3 = ax[2].scatter(range(len(imagetype_vals)), imagetype_vals-1, c=match_vals, cmap='plasma', label='Morse Index $n_j$', zorder=3)
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('$n_j$')
        ax[2].axhline(FF_res_slc[1][2]-1, color='r', linestyle='-', label='Final $n_j$', zorder=2)
        ax[2].set_title('Morse Index $n_j$')
        cbar = fig.colorbar(sc3, ax=ax[2], orientation='vertical', label='Match')
        cbar.set_label('Match')
        plt.suptitle('Convergence of Chirp Mass, Mass Ratio, Morse Index for {}  sample {}'.format(det, i))
        fig.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_SLC_convergence_{}_{}.png'.format(det, i))
        plt.close()

        ####################################################################################################################################################################        

        hf_a_ml, hf_b_ml = gen.wf_ml_fd(injection_parameters=waveform_params_a, Ml_z=0, y=0, **kwargs), gen.wf_ml_fd(injection_parameters=waveform_params_b, Ml_z=0, y=0, **kwargs)
        hf_ml = gen.pairs_ml_fd(waveform_params_a, waveform_params_b, **kwargs)
        FF_res_ml, param_history_ml = ff.compute_fitting_factor(hf_ml[det], wf_model='ML_4D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ML_4D_{}_{}.npy'.format(det, i), [FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3]])

        recovered_params_ml = waveform_params_a.copy()
        recovered_params_ml['mass_1'], recovered_params_ml['mass_2'] = mchirp_q_to_m1m2(FF_res_ml[1][0], FF_res_ml[1][1])

        hf_recovered_ml = gen.wf_ml_fd(injection_parameters = recovered_params_ml, Ml_z=FF_res_ml[1][2], y=FF_res_ml[1][3], **kwargs)
        ht_recovered_ml, ht_a_ml, ht_b_ml, ht_ml = {}, {}, {}, {}
        ht_recovered_ml[det], ht_ml[det] = hf_recovered_ml[det].to_timeseries(), hf_ml[det].to_timeseries()
        ht_a_ml[det], ht_b_ml[det] = hf_a_ml[det].to_timeseries(), hf_b_ml[det].to_timeseries()
        
        ht_ml[det].start_time, ht_a_ml[det].start_time, ht_b_ml[det].start_time = ht[det].start_time, ht_a[det].start_time, ht_b[det].start_time
        ht_recovered_ml[det].start_time += ht_ml[det].sample_times[np.argmax(np.abs(ht_ml[det]))] - ht_recovered_ml[det].sample_times[np.argmax(np.abs(ht_recovered_ml[det]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a_ml[det].sample_times, ht_a_ml[det], 'b--', label='Singles $h_A$')
        ax.plot(ht_b_ml[det].sample_times, ht_b_ml[det], 'r--', label='Singles $h_B$')
        ax.plot(ht_b_ml[det].sample_times, ht_b_ml[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_ml[det].sample_times, ht_recovered_ml[det], 'k', label='Recovery $\\tilde{h}_{\\mathcal{M}}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_ml['geocent_time'] = ht_recovered_ml[det].sample_times[np.argmax(np.abs(ht_recovered_ml[det]))]
        ax.set_xlim(recovered_params_ml['geocent_time'] - 0.5, recovered_params_ml['geocent_time'] + 0.3)
        plt.title(r'ML ({}): Recovered Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $\mathcal{{M}}_{{\ell}}^z = {:.3f}$, $y = {:.3f}$'.format(det, FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ML_4D_{}_{}.png'.format(det, i))
        plt.close()

        param_history_ml = np.array(param_history_ml)
        chirp_mass_vals, q_vals, Ml_z_vals, y_vals, match_vals = param_history_ml[:, 0], param_history_ml[:, 1], param_history_ml[:, 2], param_history_ml[:, 3], param_history_ml[:, 4]

        fig, ax = plt.subplots(2, 2, figsize=(13, 12))
        sc1 = ax[0, 0].scatter(range(len(chirp_mass_vals)), chirp_mass_vals, c=match_vals, cmap='plasma', label='Chirp Mass $\\mathcal{M}$', zorder=3)
        ax[0, 0].set_xlabel('Iteration')
        ax[0, 0].set_ylabel('$\\mathcal{M}$')
        ax[0, 0].axhline(waveform_params_a['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_A$', zorder=1)
        ax[0, 0].axhline(waveform_params_b['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_B$', zorder=1)
        ax[0, 0].axhline(FF_res_ml[1][0], color='r', linestyle='-', label='Final $\\mathcal{M}$', zorder=2)
        ax[0, 0].set_title('Chirp Mass $\\mathcal{M}$')
        sc2 = ax[0, 1].scatter(range(len(q_vals)), q_vals, c=match_vals, cmap='plasma', label='Mass Ratio $q$', zorder=3)
        ax[0, 1].set_xlabel('Iteration')
        ax[0, 1].set_ylabel('$q$')
        ax[0, 1].axhline(waveform_params_a['mass_ratio'], color='k', linestyle='--', label='$q_A$', zorder=1)
        ax[0, 1].axhline(waveform_params_b['mass_ratio'], color='k', linestyle='--', label='$q_B$', zorder=1)
        ax[0, 1].axhline(FF_res_ml[1][1], color='r', linestyle='-', label='Final $q$', zorder=2)
        ax[0, 1].set_title('Mass Ratio $q$')
        sc3 = ax[1, 0].scatter(range(len(Ml_z_vals)), Ml_z_vals, c=match_vals, cmap='plasma', label='Spin $\\mathcal{M}_\\ell^z$', zorder=3)
        ax[1, 0].set_xlabel('Iteration')
        ax[1, 0].set_ylabel('$\\mathcal{M}_\\ell^z$')
        ax[1, 0].axhline(FF_res_ml[1][2], color='r', linestyle='-', label='Final $\\mathcal{M}_\\ell^z$', zorder=2)
        ax[1, 0].set_title('Lens Mass $\\mathcal{M}_\\ell^z$')
        sc4 = ax[1, 1].scatter(range(len(y_vals)), y_vals, c=match_vals, cmap='plasma', label='Spin $y$', zorder=3)
        ax[1, 1].set_xlabel('Iteration')
        ax[1, 1].set_ylabel('$y$')
        ax[1, 1].axhline(FF_res_ml[1][3], color='r', linestyle='-', label='Final $y$', zorder=2)
        ax[1, 1].set_title('Impact Parameter $y$')
        cbar1 = fig.colorbar(sc2, ax=ax[0, 1], orientation='vertical', label='Match')
        cbar1.set_label('Match')
        cbar2 = fig.colorbar(sc4, ax=ax[1, 1], orientation='vertical', label='Match')
        cbar2.set_label('Match')
        plt.suptitle('Convergence of Chirp Mass, Mass Ratio, Lens Mass, Impact Parameter for {}  sample {}$'.format(det, i))
        fig.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ML_convergence_{}_{}.png'.format(det, i))
        plt.close()

        ####################################################################################################################################################################        

        ht_ecc_0 =  gen.pairs_ecc_td(waveform_params_a, waveform_params_b, **kwargs)
        FF_res_ecc_0, param_history_ecc_0 = ff.compute_fitting_factor(ht_ecc_0[det], wf_model='EC_0_2D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ECC_0_2D_{}_{}.npy'.format(det, i), [FF_res_ecc_0[0], FF_res_ecc_0[1][0], FF_res_ecc_0[1][1]])

        recovered_params_ecc_0 = waveform_params_a.copy()
        recovered_params_ecc_0['mass_1'], recovered_params_ecc_0['mass_2'] = mchirp_q_to_m1m2(FF_res_ecc_0[1][0], FF_res_ecc_0[1][1])

        ht_recovered_ecc_0 = gen.wf_ecc_td(injection_parameters = recovered_params_ecc_0, e=0, **kwargs)
        ht_recovered_ecc_0[det].start_time += ht_ecc_0[det].sample_times[np.argmax(np.abs(ht_ecc_0[det]))] - ht_recovered_ecc_0[det].sample_times[np.argmax(np.abs(ht_recovered_ecc_0[det]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_ecc_0[det].sample_times, ht_ecc_0[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_ecc_0[det].sample_times, ht_recovered_ecc_0[det], 'k', label='Recovery $\\tilde{h}_{e}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_ecc_0['geocent_time'] = ht_recovered_ecc_0[det].sample_times[np.argmax(np.abs(ht_recovered_ecc_0[det]))]
        ax.set_xlim(recovered_params_ecc_0['geocent_time'] - 0.5, recovered_params_ecc_0['geocent_time'] + 0.3)
        plt.title(r'ECC_0 ({}): Recovered Waveform with Match: ${:.3f}$ and 3D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(det, FF_res_ecc_0[0], FF_res_ecc_0[1][0], FF_res_ecc_0[1][1]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ECC_0_2D_{}_{}.png'.format(det, i))
        plt.close()

        param_history_ecc_0 = np.array(param_history_ecc_0)
        chirp_mass_vals, q_vals, match_vals = param_history_ecc_0[:, 0], param_history_ecc_0[:, 1], param_history_ecc_0[:, 2]

        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sc1 = ax[0].scatter(range(len(chirp_mass_vals)), chirp_mass_vals, c=match_vals, cmap='plasma', label='Chirp Mass $\\mathcal{M}$', zorder=3)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('$\\mathcal{M}$')
        ax[0].axhline(waveform_params_a['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_A$', zorder=1)
        ax[0].axhline(waveform_params_b['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_B$', zorder=1)
        ax[0].axhline(FF_res[1][0], color='r', linestyle='-', label='Final $\\mathcal{M}$', zorder=2)
        ax[0].set_title('Chirp Mass $\\mathcal{M}$')
        sc2 = ax[1].scatter(range(len(q_vals)), q_vals, c=match_vals, cmap='plasma', label='Mass Ratio $q$', zorder=3)
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('$q$')
        ax[1].axhline(waveform_params_a['mass_ratio'], color='k', linestyle='--', label='$q_A$', zorder=1)
        ax[1].axhline(waveform_params_b['mass_ratio'], color='k', linestyle='--', label='$q_B$', zorder=1)
        ax[1].axhline(FF_res[1][1], color='r', linestyle='-', label='Final $q$', zorder=2)
        ax[1].set_title('Mass Ratio $q$')
        cbar = fig.colorbar(sc2, ax=ax[1], orientation='vertical', label='Match')
        cbar.set_label('Match')
        plt.suptitle('Convergence of Chirp Mass, Mass Ratio for {} sample {}'.format(det, i))
        fig.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ECC_0_convergence_{}_{}.png'.format(det, i))
        plt.close()

        ####################################################################################################################################################################        

        ht_ecc =  gen.pairs_ecc_td(waveform_params_a, waveform_params_b, **kwargs)
        FF_res_ecc, param_history_ecc = ff.compute_fitting_factor(ht_ecc[det], wf_model='EC_3D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ECC_3D_{}_{}.npy'.format(det, i), [FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2]])

        recovered_params_ecc = waveform_params_a.copy()
        recovered_params_ecc['mass_1'], recovered_params_ecc['mass_2'] = mchirp_q_to_m1m2(FF_res_ecc[1][0], FF_res_ecc[1][1])

        ht_recovered_ecc = gen.wf_ecc_td(injection_parameters = recovered_params_ecc, e=FF_res_ecc[1][2], **kwargs)
        ht_recovered_ecc[det].start_time += ht_ecc[det].sample_times[np.argmax(np.abs(ht_ecc[det]))] - ht_recovered_ecc[det].sample_times[np.argmax(np.abs(ht_recovered_ecc[det]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_a[det].sample_times, ht_a[det], 'r--', label='Singles $h_A$')
        ax.plot(ht_b[det].sample_times, ht_b[det], 'b--', label='Singles $h_B$')
        ax.plot(ht_ecc[det].sample_times, ht_ecc[det], 'm', label='Pairs $h_A + h_B$')
        ax.plot(ht_recovered_ecc[det].sample_times, ht_recovered_ecc[det], 'k', label='Recovery $\\tilde{h}_{e}$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_ecc['geocent_time'] = ht_recovered_ecc[det].sample_times[np.argmax(np.abs(ht_recovered_ecc[det]))]
        ax.set_xlim(recovered_params_ecc['geocent_time'] - 0.5, recovered_params_ecc['geocent_time'] + 0.3)
        plt.title(r'ECC ({}): Recovered Waveform with Match: ${:.3f}$ and 3D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$'.format(det, FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ECC_3D_{}_{}.png'.format(det, i))
        plt.close()

        param_history_ecc = np.array(param_history_ecc)
        chirp_mass_vals, q_vals, ecc_vals, match_vals = param_history_ecc[:, 0], param_history_ecc[:, 1], param_history_ecc[:, 2], param_history_ecc[:, 3]

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        sc1 = ax[0].scatter(range(len(chirp_mass_vals)), chirp_mass_vals, c=match_vals, cmap='plasma', label='Chirp Mass $\\mathcal{M}$', zorder=3)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('$\\mathcal{M}$')
        ax[0].axhline(waveform_params_a['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_A$', zorder=1)
        ax[0].axhline(waveform_params_b['chirp_mass'], color='k', linestyle='--', label='$\\mathcal{M}_B$', zorder=1)
        ax[0].axhline(FF_res_ecc[1][0], color='r', linestyle='-', label='Final $\\mathcal{M}$', zorder=2)
        ax[0].set_title('Chirp Mass $\\mathcal{M}$')
        sc2 = ax[1].scatter(range(len(q_vals)), q_vals, c=match_vals, cmap='plasma', label='Mass Ratio $q$', zorder=3)
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('$q$')
        ax[1].axhline(waveform_params_a['mass_ratio'], color='k', linestyle='--', label='$q_A$', zorder=1)
        ax[1].axhline(waveform_params_b['mass_ratio'], color='k', linestyle='--', label='$q_B$', zorder=1)
        ax[1].axhline(FF_res_ecc[1][1], color='r', linestyle='-', label='Final $q$', zorder=2)
        ax[1].set_title('Mass Ratio $q$')
        sc3 = ax[2].scatter(range(len(ecc_vals)), ecc_vals, c=match_vals, cmap='plasma', label='Eccentricity $e$', zorder=3)
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('$e$')
        ax[2].axhline(FF_res_ecc[1][2], color='r', linestyle='-', label='Final $e$', zorder=2)
        ax[2].set_title('Eccentricity $e$')
        cbar = fig.colorbar(sc3, ax=ax[2], orientation='vertical', label='Match')
        cbar.set_label('Match')
        plt.suptitle('Convergence of Chirp Mass, Mass Ratio, Eccentricity for {} sample {}$'.format(det, i))
        fig.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population_3G/population/FF_ECC_convergence_{}_{}.png'.format(det, i))
        plt.close()

        #################################################################################################################################################################### 