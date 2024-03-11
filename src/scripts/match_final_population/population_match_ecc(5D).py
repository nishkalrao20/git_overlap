#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python
# 5 Dimensional Eccentric (Chirp mass, Mass ratio, Spin1z, Spin2z, e) Nelder Mead maximization of Match on PAIRS waveforms

import os
import pickle
import numpy as np
from tqdm import tqdm

import sys 
sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import FF_computation as ff 
import waveforms as wf

gen = wf.PairsWaveformGeneration()

f_lower = 20.0
f_ref = 50.0
sampling_frequency = 4096.0
f_high = sampling_frequency / 2.0
delta_t = 1.0 / sampling_frequency

kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, f_high=f_high, delta_t=delta_t)

files = os.listdir('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections')

file_indices = []
for file in files:
    if file.endswith('.pkl') and 'GW Waveform A Meta Data' in file:
        index_start = file.find('GW Waveform A Meta Data') + len('GW Waveform A Meta Data ')
        index_end = file.find('.pkl')
        file_index = file[index_start:index_end]
        file_indices.append(int(file_index))
N_sampl = len(file_indices)
queue = 5000
k = int(sys.argv[1])
indices = file_indices[k*int(N_sampl/queue):(k+1)*int(N_sampl/queue)]

keys = ['mass_1_source', 'mass_ratio', 'a_1', 'a_2', 'redshift', 'cos_tilt_1', 'cos_tilt_2', 'phi_12', 'phi_jl', 'cos_theta_jn', 'ra', 'dec', 'psi', 'phase', 'incl', 'cos_theta_zn', 'mass_1', 'mass_2', 'luminosity_distance', 'tilt_1', 'tilt_2', 'theta_jn', 'theta_zn', 'geocent_time', 'snr_det']

data_a, data_b = {key: np.zeros(len(indices)) for key in keys}, {key: np.zeros(len(indices)) for key in keys}
waveform_metadata_a, waveform_metadata_b = [], []

for k, idx in enumerate(indices):
    
    waveform_metadata_a.append(pickle.load(open('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections/GW Waveform A Meta Data %s.pkl' % idx, 'rb')))   # Importing Waveform Meta Data
    for key, val in waveform_metadata_a[k]['H1']['parameters'].items():   # Setting the variables
        if key in data_a:
            data_a[key][k] = val

    waveform_metadata_b.append(pickle.load(open('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections/GW Waveform B Meta Data %s.pkl' % idx, 'rb')))   # Importing Waveform Meta Data
    for key, val in waveform_metadata_b[k]['H1']['parameters'].items():   # Setting the variables
        if key in data_b:
            data_b[key][k] = val

mchirp_a, mchirp_b = np.power(data_a['mass_1']*data_a['mass_2'], (3/5))/np.power(data_a['mass_1']+data_a['mass_2'], (1/5)), np.power(data_b['mass_1']*data_b['mass_2'], (3/5))/np.power(data_b['mass_1']+data_b['mass_2'], (1/5))
eta_a, eta_b = (data_a['mass_1']*data_a['mass_2'])/np.power(data_a['mass_1']+data_a['mass_2'], 2), (data_b['mass_1']*data_b['mass_2'])/np.power(data_b['mass_1']+data_b['mass_2'], 2)
eff_spin_a, eff_spin_b = data_a['a_1'], data_b['a_1']

delta_tc = data_b['geocent_time'] - data_a['geocent_time']
snr_a, snr_b = data_a['snr_det'], data_b['snr_det']

# Evaluating the FF values for the PAIRS
for k, idx in enumerate(indices):

    ht = gen.pairs_ecc_td(injection_parameters_a = waveform_metadata_a[k]['H1']['parameters'], injection_parameters_b = waveform_metadata_b[k]['H1']['parameters'], **kwargs)
    FF_res = ff.compute_fitting_factor(ht['H1'], wf_model='EC_5D', apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                    n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
    
    recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_e = FF_res[1]
    
    np.savetxt('/home/nishkal.rao/git_overlap/src/output/match_final_population/outputs_ecc/PAIRS(5D) %s.csv'%(idx), np.column_stack((FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_e)), delimiter=',')