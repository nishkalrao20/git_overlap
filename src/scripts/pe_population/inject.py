import numpy as np
from tqdm import tqdm
import itertools
from pesummary.io import read
from pycbc.frame import write_frame
from pycbc import filter
import pycbc.psd
import gwmat
import matplotlib.pyplot as plt

f_lower = 20.0
f_ref = 50.0
duration = 32.0
sampling_frequency = 4096.0
delta_t = 1.0 / sampling_frequency
f_higher = sampling_frequency / 2.0

kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, delta_t=delta_t)
mchirpratio, snrratio = [.5, 1., 2.], [.5, 1.]
delta_tc = [-.1, -.05, -.03, -.02, -.01, .01, .02, .03, .05, .1]

combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))
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
default_kwargs = {"wf_domain":"TD", "approximant":"IMRPhenomXPHM", "rwrap":-0.2, "delta_t":1/sampling_frequency, 
                      "taper_hp_hc":True, "hp_hc_extra_padding_at_start":8}
default_kwargs.update(kwargs)
kwargs = default_kwargs.copy()
waveform_params_a.update(kwargs)
waveform_params_b.update(kwargs)
injections = {}
for i in tqdm(range(len(combinations))):

    waveform_params_c = waveform_params_b.copy()
    waveform_params_c['mass_1'] = waveform_params_a['mass_1']*combinations[i][0]
    waveform_params_c['mass_2'] = waveform_params_a['mass_2']*combinations[i][0]
    waveform_params_c['geocent_time'] = waveform_params_a['geocent_time'] + combinations[i][2]
    
    wfs_res_1 = gwmat.injection.simulate_injection_with_comprehensive_output(**waveform_params_a)
    wfs_res_2 = gwmat.injection.simulate_injection_with_comprehensive_output(**waveform_params_c)

    waveform_params_a['snr_det'], waveform_params_c['snr_det'] = wfs_res_1['network_matched_filter_snr'], wfs_res_2['network_matched_filter_snr']
    waveform_params_a['luminosity_distance'] = waveform_params_a['luminosity_distance']*waveform_params_a['snr_det']/30  
    waveform_params_c['luminosity_distance'] = waveform_params_c['luminosity_distance']*waveform_params_c['snr_det']/(30*combinations[i][1])    

    wfs_res_1 = gwmat.injection.simulate_injection_with_comprehensive_output(**waveform_params_a)
    wfs_res_2 = gwmat.injection.simulate_injection_with_comprehensive_output(**waveform_params_c)

    waveform_params_a['snr_det'], waveform_params_c['snr_det'] = wfs_res_1['network_matched_filter_snr'], wfs_res_2['network_matched_filter_snr']

    ht_a, ht_b, ht = {}, {}, {}
    for ifo in wfs_res_1["pure_ifo_signal"].keys():
        ht_a[ifo] = wfs_res_1["pure_ifo_signal"][ifo]
        ht_b[ifo] = wfs_res_2["pure_ifo_signal"][ifo]
        dt = float(ht_b[ifo].start_time) - float(ht_a[ifo].start_time)
        if dt > 0: 
            ht_b[ifo] = gwmat.injection.modify_signal_start_time(ht_b[ifo], extra = dt)
        if dt < 0: 
            ht_a[ifo] = gwmat.injection.modify_signal_start_time(ht_a[ifo], extra = abs(dt))    
        ht[ifo] = ht_a[ifo] + ht_b[ifo]

    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/SINGLES_A_{}_{}_{}.npy'.format(combinations[i][0], combinations[i][1], combinations[i][2]), waveform_params_a)
    np.save('/home/nishkal.rao/git_overlap/src/output/pe_population/injections/SINGLES_B_{}_{}_{}.npy'.format(combinations[i][0], combinations[i][1], combinations[i][2]), waveform_params_c)

    for idx, det in enumerate(['H1', 'L1', 'V1']):
        outfile = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/PAIRS_{}_{}_{}_{}.gwf'.format(det, combinations[i][0], combinations[i][1], combinations[i][2])
        write_frame(outfile, "{}:PyCBC_Injection".format(det), ht[det])