import os
import pickle
import deepdish
import numpy as np
import pandas as pd

files = os.listdir('git_overlap/src/output/match_population/match_population_non_spinning/injections')

file_indices_a, file_indices_b = [], []
for file in files:
    if file.endswith('.pkl') and 'GW Waveform A Meta Data' in file:
        index_start = file.find('GW Waveform A Meta Data') + len('GW Waveform A Meta Data ')
        index_end = file.find('.pkl')
        file_index = file[index_start:index_end]
        file_indices_a.append(int(file_index))
    if file.endswith('.pkl') and 'GW Waveform B Meta Data' in file:
        index_start = file.find('GW Waveform B Meta Data') + len('GW Waveform B Meta Data ')
        index_end = file.find('.pkl')
        file_index = file[index_start:index_end]
        file_indices_b.append(int(file_index))

N_sampl = len(file_indices_a)
waveform_metadata_a, waveform_metadata_b = [], []
indices = np.random.choice(file_indices_a, N_sampl, replace=False)
    
injection = dict(deepdish.io.load('git_overlap/src/output/injections/injections.hdf5')['injections'])

for key, val in injection.items():
    exec(key + '=val')

for k in range(N_sampl):
    
    eff_spin_a = (a_1[k] * mass_1[k] + a_2[k] * mass_2[k]) / (mass_1[k] + mass_2[k])
    eff_spin_b = (a_1[N_sampl-k] * mass_1[N_sampl-k] + a_2[N_sampl-k] * mass_2[N_sampl-k]) / (mass_1[N_sampl-k] + mass_2[N_sampl-k])

    waveform_metadata_a = pickle.load(open('git_overlap/src/output/match_population/match_population_non_spinning/injections/GW Waveform A Meta Data {}.pkl'.format(indices[k]), 'rb'))
    waveform_metadata_a['H1']['parameters']['a_1'], waveform_metadata_a['H1']['parameters']['a_2'] = eff_spin_a, eff_spin_a
    waveform_metadata_a['L1']['parameters']['a_1'], waveform_metadata_a['L1']['parameters']['a_2'] = eff_spin_a, eff_spin_a
    waveform_metadata_a['V1']['parameters']['a_1'], waveform_metadata_a['V1']['parameters']['a_2'] = eff_spin_a, eff_spin_a
    with open('git_overlap/src/output/match_population/match_population_equal_spins/injections/GW Waveform A Meta Data {}.pkl'.format(indices[k]), 'wb') as file:
        pickle.dump(waveform_metadata_a, file)

    waveform_metadata_b = pickle.load(open('git_overlap/src/output/match_population/match_population_non_spinning/injections/GW Waveform B Meta Data {}.pkl'.format(indices[k]), 'rb'))
    waveform_metadata_b['H1']['parameters']['a_1'], waveform_metadata_b['H1']['parameters']['a_2'] = eff_spin_b, eff_spin_b
    waveform_metadata_b['L1']['parameters']['a_1'], waveform_metadata_b['L1']['parameters']['a_2'] = eff_spin_b, eff_spin_b
    waveform_metadata_b['V1']['parameters']['a_1'], waveform_metadata_b['V1']['parameters']['a_2'] = eff_spin_b, eff_spin_b
    with open('git_overlap/src/output/match_population/match_population_equal_spins/injections/GW Waveform B Meta Data {}.pkl'.format(indices[k]), 'wb') as file:
        pickle.dump(waveform_metadata_b, file)