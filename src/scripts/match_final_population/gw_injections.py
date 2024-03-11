#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# Generate Overlapping BBH Injections from a population and inject into detectors (H1, L1, V1)

import bilby
import pickle
import deepdish
import gwpopulation
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from pycbc import filter

import sys 
sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import waveforms as wf
import FF_computation as ff

gen = wf.PairsWaveformGeneration()

# Setting some constants

duration = 100.0
minimum_frequency = 20.0
reference_frequency = 50.0
sampling_frequency = 4096.0
delta_t = 1.0 / sampling_frequency
maximum_frequency = sampling_frequency / 2.0
kwargs = dict(sampling_frequency=sampling_frequency, f_lower=minimum_frequency, f_ref=maximum_frequency, delta_t=delta_t)

injection = dict(deepdish.io.load('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections/injections.hdf5')['injections'])
mchirp = (injection['mass_1'] * injection['mass_2']) ** (3 / 5) / (injection['mass_1'] + injection['mass_2']) ** (1 / 5)

num_points = int(len(mchirp) / 2)
idxs = np.random.choice(len(mchirp), size=(num_points, 2), replace=False)
unique_idxs, counts = np.unique(idxs[:, 0], return_counts=True)
mask = counts == 1
filtered_idxs = idxs[mask]

delta_tc = sp.stats.truncnorm.rvs(0,4/3, loc=0, scale=.75, size=int(len(filtered_idxs)/2))
delta_tc = np.concatenate((delta_tc, -delta_tc))

k, queue = int(sys.argv[1]), 1000
filtered_idxs_mod = filtered_idxs[k*int(len(filtered_idxs)/queue):(k+1)*int(len(filtered_idxs)/queue)]

for i, index in enumerate(tqdm(filtered_idxs_mod)):

    mchirp_ratio = mchirp[index[1]] / mchirp[index[0]]

    if 0.5 <= mchirp_ratio <= 2:

        injection_parameters_a, injection_parameters_b = {}, {}
        
        for key, val in injection.items():
            injection_parameters_a[key], injection_parameters_b[key] = val[index[0]], val[index[1]]

        injection_parameters_b['geocent_time'] += delta_tc[np.random.randint(0, len(delta_tc))]

        start_time_a = injection_parameters_a['geocent_time'] - duration + 2
        start_time_b = injection_parameters_b['geocent_time'] - duration + 2
                
        ht_a, ht_b = gen.wf_td(injection_parameters_a, **kwargs), gen.wf_td(injection_parameters_b, **kwargs)
        ht = gen.pairs_td(injection_parameters_a, injection_parameters_b, **kwargs)

        snr_a, snr_b, snr = np.zeros(3), np.zeros(3), np.zeros(3)
        for idx, det in enumerate(['H1', 'L1', 'V1']):
            snr_a[idx] = filter.matchedfilter.sigma(ht_a[det], psd=None, low_frequency_cutoff=minimum_frequency, high_frequency_cutoff=maximum_frequency)
            snr_b[idx] = filter.matchedfilter.sigma(ht_b[det], psd=None, low_frequency_cutoff=minimum_frequency, high_frequency_cutoff=maximum_frequency)

        injection_parameters_a['snr_det'], injection_parameters_b['snr_det'] = np.sqrt(np.sum(snr_a**2)), np.sqrt(np.sum(snr_b**2))

        if 0.5 <= injection_parameters_b['snr_det']/injection_parameters_a['snr_det'] <= 2:

            waveform_generator_a = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency, start_time=start_time_a,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency,
                                    'minimum_frequency': minimum_frequency}
            )

            waveform_generator_b = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency, start_time=start_time_b,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency,
                                    'minimum_frequency': minimum_frequency}
            )

            ifos_a, ifos_b, ifos_c = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

            for det in [ifos_a, ifos_b, ifos_c]:
                for ifo in det:
                    ifo.minimum_frequency = minimum_frequency
                    ifo.maximum_frequency = sampling_frequency/2
                det.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=start_time_a)

            ifos_a.inject_signal(waveform_generator=waveform_generator_a, parameters=injection_parameters_a)
            ifos_b.inject_signal(waveform_generator=waveform_generator_b, parameters=injection_parameters_b)

            ifos_c.inject_signal(waveform_generator=waveform_generator_a, parameters=injection_parameters_a)
            ifos_c.inject_signal(waveform_generator=waveform_generator_b, parameters=injection_parameters_b)

            if (np.abs(ifos_c[0].meta_data['matched_filter_SNR']))**2+(np.abs(ifos_c[1].meta_data['matched_filter_SNR']))**2+(np.abs(ifos_c[2].meta_data['matched_filter_SNR']))**2 > 8:
                with open('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections/GW Waveform A Meta Data %s.pkl'%(index[0]), 'wb') as file:
                    pickle.dump(ifos_a.meta_data, file)
                with open('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections/GW Waveform B Meta Data %s.pkl'%(index[0]), 'wb') as file:
                    pickle.dump(ifos_b.meta_data, file)