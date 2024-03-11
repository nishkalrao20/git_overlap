#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# Generate Overlapping BBH Injections from a population and inject into detectors (H1, L1, V1)

import numpy as np
from tqdm import tqdm
import itertools
from pesummary.io import read
from pycbc.frame import write_frame
from pycbc import filter

import sys 
sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import waveforms as wf
import FF_computation as ff
from FF_computation import mchirp_q_to_m1m2

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

mchirpratio, snrratio = [.5, 1., 2.], [.5, 1.]
delta_tc = [-.1, -.05, -.03, -.02, -.01, .01, .02, .03, .05, .1]

combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))

# Parameters for the injection: waveform_params_a: GW150914, waveform_params_c: GW170814

injection_a = read('/home/nishkal.rao/git_overlap/src/data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']   # Loading the GW150914 Posterior distributions
injection_b = read('/home/nishkal.rao/git_overlap/src/data/IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']   # Loading the GW170814 Posterior distributions

nmap_a = np.argmax(injection_a['log_likelihood']+injection_a['log_prior'])   # Maximum A Posteriori values
nmap_b = np.argmax(injection_b['log_likelihood']+injection_b['log_prior'])   # Maximum A Posteriori values

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

N, queue = len(combinations), 60
k = int(sys.argv[1])
combinations = combinations[k*int(N/queue):(k+1)*int(N/queue)]

injections = {}
for i in tqdm(range(len(combinations))):

    waveform_params_c = waveform_params_b.copy()
    waveform_params_c['mass_1'] = waveform_params_a['mass_1']*combinations[i][0]
    waveform_params_c['mass_2'] = waveform_params_a['mass_2']*combinations[i][0]
    waveform_params_c['geocent_time'] = waveform_params_a['geocent_time'] + combinations[i][2]

    ht_a, ht_b = gen.wf_td(waveform_params_a, **kwargs), gen.wf_td(waveform_params_c, **kwargs)

    snr_a, snr_b = np.zeros(3), np.zeros(3)
    for idx, det in enumerate(['H1', 'L1', 'V1']):
        snr_a[idx] = filter.matchedfilter.sigma(ht_a[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        snr_b[idx] = filter.matchedfilter.sigma(ht_b[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
    waveform_params_a['snr_det'], waveform_params_c['snr_det'] = np.sqrt(np.sum(snr_a**2)), np.sqrt(np.sum(snr_b**2))

    waveform_params_c['luminosity_distance'] = waveform_params_c['luminosity_distance']*waveform_params_c['snr_det']/(waveform_params_a['snr_det']*combinations[i][1])    
    ht_a, ht_b = gen.wf_td(waveform_params_a, **kwargs), gen.wf_td(waveform_params_c, **kwargs)
    for idx, det in enumerate(['H1', 'L1', 'V1']):
        snr_a[idx] = filter.matchedfilter.sigma(ht_a[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)
        snr_b[idx] = filter.matchedfilter.sigma(ht_b[det], psd=None, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_higher)  
    waveform_params_a['snr_det'], waveform_params_c['snr_det'] = np.sqrt(np.sum(snr_a**2)), np.sqrt(np.sum(snr_b**2))

    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/SINGLES_A_{}_{}_{}.npy'.format(combinations[i][0], combinations[i][1], combinations[i][2]), waveform_params_a)
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/SINGLES_B_{}_{}_{}.npy'.format(combinations[i][0], combinations[i][1], combinations[i][2]), waveform_params_c)

    for idx, det in enumerate(['H1', 'L1', 'V1']):
        ht = gen.pairs_td(waveform_params_a, waveform_params_c, **kwargs)
        outfile = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/PAIRS_{}_{}_{}_{}.gwf'.format(det, combinations[i][0], combinations[i][1], combinations[i][2])
        write_frame(outfile, "{}:PyCBC_Injection".format(det), ht[det])

    for idx, det in enumerate(['H1', 'L1', 'V1']):
        ht = gen.pairs_td(waveform_params_a, waveform_params_c, **kwargs)
        FF_res = ff.compute_fitting_factor(ht[det], wf_model='4D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_4D_{}_{}_{}_{}.npy'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]), [FF_res[0], FF_res[1][0], FF_res[1][1], FF_res[1][2], FF_res[1][3]])
        
        recovered_params = waveform_params_a.copy()
        recovered_params['mass_1'], recovered_params['mass_2'] = mchirp_q_to_m1m2(FF_res[1][0], FF_res[1][1])
        recovered_params['spin1z'], recovered_params['spin2z'] = FF_res[1][2], FF_res[1][3]

        ht_recovered = gen.wf_td(injection_parameters = recovered_params, **kwargs)
        for key in ht.keys():
            ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
            ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht[det].sample_times, ht[det], label='Injected Waveform')
        ax.plot(ht_recovered[det].sample_times, ht_recovered[det], label='Recovered Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        recovered_params['geocent_time'] = ht_recovered[det].sample_times[np.argmax(np.abs(ht_recovered[det]))]        
        ax.set_xlim(recovered_params['geocent_time'] - 0.5, recovered_params['geocent_time'] + 0.3)
        plt.title(r'SINGLES ({}): Recovered Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$'.format(det, FF_res[0], FF_res[1][0], FF_res[1][1], FF_res[1][2], FF_res[1][3]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_4D_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()

        ####################################################################################################################################################################        

        ht_ecc =  gen.pairs_ecc_td(waveform_params_a, waveform_params_c, **kwargs)
        FF_res_ecc = ff.compute_fitting_factor(ht_ecc[det], wf_model='EC_5D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_ECC_5D_{}_{}_{}_{}.npy'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]), [FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2], FF_res_ecc[1][3], FF_res_ecc[1][4]])

        recovered_params_ecc = waveform_params_a.copy()
        recovered_params_ecc['mass_1'], recovered_params_ecc['mass_2'] = mchirp_q_to_m1m2(FF_res_ecc[1][0], FF_res_ecc[1][1])
        recovered_params_ecc['spin1z'], recovered_params_ecc['spin2z'] = FF_res_ecc[1][2], FF_res_ecc[1][3]

        ht_recovered_ecc = gen.wf_ecc_td(injection_parameters = recovered_params_ecc, e=FF_res_ecc[1][4], **kwargs)
        for key in ht_ecc.keys():
            ht_ecc[key].start_time += ht_recovered_ecc[key].sample_times[np.argmax(np.abs(ht_recovered_ecc[key]))] - ht_ecc[key].sample_times[np.argmax(np.abs(ht_ecc[key]))]
            ht_recovered_ecc[key].start_time += ht_ecc[key].sample_times[np.argmax(np.abs(ht_ecc[key]))] - ht_recovered_ecc[key].sample_times[np.argmax(np.abs(ht_recovered_ecc[key]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_ecc[det].sample_times, ht_ecc[det], label='Injected Waveform')
        ax.plot(ht_recovered_ecc[det].sample_times, ht_recovered_ecc[det], label='Recovered Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_ecc['geocent_time'] = ht_recovered_ecc[det].sample_times[np.argmax(np.abs(ht_recovered_ecc[det]))]
        ax.set_xlim(recovered_params_ecc['geocent_time'] - 0.5, recovered_params_ecc['geocent_time'] + 0.3)
        plt.title(r'ECC ({}): Recovered Waveform with Match: ${:.3f}$ and 5D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$, $e = {:.3f}$'.format(det, FF_res_ecc[0], FF_res_ecc[1][0], FF_res_ecc[1][1], FF_res_ecc[1][2], FF_res_ecc[1][3], FF_res_ecc[1][4]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_ECC_5D_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()

        ####################################################################################################################################################################

        hf_ml = gen.pairs_ml_fd(waveform_params_a, waveform_params_c, **kwargs)
        FF_res_ml = ff.compute_fitting_factor(hf_ml[det], wf_model='ML_6D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_higher, psd=None,
                                        n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
        np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_ML_6D_{}_{}_{}_{}.npy'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]), [FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3], FF_res_ml[1][4], FF_res_ml[1][5]])

        recovered_params_ml = waveform_params_a.copy()
        recovered_params_ml['mass_1'], recovered_params_ml['mass_2'] = mchirp_q_to_m1m2(FF_res_ml[1][0], FF_res_ml[1][1])
        recovered_params_ml['spin1z'], recovered_params_ml['spin2z'] = FF_res_ml[1][2], FF_res_ml[1][3]

        hf_recovered_ml = gen.wf_ml_fd(injection_parameters = recovered_params_ml, Ml_z=FF_res_ml[1][4], y=FF_res_ml[1][5], **kwargs)
        ht_recovered_ml = {'H1': hf_recovered_ml['H1'].to_timeseries(), 'L1': hf_recovered_ml['L1'].to_timeseries(), 'V1': hf_recovered_ml['V1'].to_timeseries()}
        ht_ml = {'H1': hf_ml['H1'].to_timeseries(), 'L1': hf_ml['L1'].to_timeseries(), 'V1': hf_ml['V1'].to_timeseries()}
        for key in ht_ml.keys():
            ht_ml[key].start_time += ht_recovered_ml[key].sample_times[np.argmax(np.abs(ht_recovered_ml[key]))] - ht_ml[key].sample_times[np.argmax(np.abs(ht_ml[key]))]
            ht_recovered_ml[key].start_time += ht_ml[key].sample_times[np.argmax(np.abs(ht_ml[key]))] - ht_recovered_ml[key].sample_times[np.argmax(np.abs(ht_recovered_ml[key]))]

        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        ax.plot(ht_ml[det].sample_times, ht_ml[det], label='Injected Waveform')
        ax.plot(ht_recovered_ml[det].sample_times, ht_recovered_ml[det], label='Recovered Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.legend()
        recovered_params_ml['geocent_time'] = ht_recovered_ml[det].sample_times[np.argmax(np.abs(ht_recovered_ml[det]))]
        ax.set_xlim(recovered_params_ml['geocent_time'] - 0.5, recovered_params_ml['geocent_time'] + 0.3)
        plt.title(r'ML ({}): Recovered Waveform with Match: ${:.3f}$ and 6D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$, $\mathcal{{M}}_{{\ell}}^z = {:.3f}$, $y = {:.3f}$'.format(det, FF_res_ml[0], FF_res_ml[1][0], FF_res_ml[1][1], FF_res_ml[1][2], FF_res_ml[1][3], FF_res_ml[1][4], FF_res_ml[1][5]))
        plt.tight_layout()
        plt.savefig('/home/nishkal.rao/git_overlap/src/output/pe_population/FF/FF_ML_6D_{}_{}_{}_{}.png'.format(det, combinations[i][0], combinations[i][1], combinations[i][2]))
        plt.close()