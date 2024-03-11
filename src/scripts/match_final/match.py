import numpy as np
import pickle

import sys 
sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')
import FF_computation as ff 
import waveforms as wf
from FF_computation import mchirp_q_to_m1m2

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

gen = wf.PairsWaveformGeneration()

f_lower = 20.0
f_ref = 50.0
sampling_frequency = 4096.0
f_high = sampling_frequency / 2.0
delta_t = 1.0 / sampling_frequency

kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, f_high=f_high, delta_t=delta_t)

waveform_metadata_a, waveform_metadata_b = pickle.load(open('/home/nishkal.rao/git_overlap/src/output/injections/GW Waveform A Meta Data.pkl', 'rb')), pickle.load(open('/home/nishkal.rao/git_overlap/src/output/injections/GW Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data
print('Injection Parameters A: ', waveform_metadata_a['H1']['parameters'])
print('Injection Parameters B: ', waveform_metadata_b['H1']['parameters'])

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 2D Fitting Factor for SINGLES')
ht = gen.wf_td(injection_parameters = waveform_metadata_a['H1']['parameters'], **kwargs)
wf_model = "2D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

ht_recovered = gen.wf_td(injection_parameters = recovered_injection_parameters, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'SINGLESvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 2D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - SINGLESvSINGLES (2D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 2D Fitting Factor for PAIRS')
ht = gen.pairs_td(injection_parameters_a = waveform_metadata_a['H1']['parameters'], injection_parameters_b = waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "2D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

ht_recovered = gen.wf_td(injection_parameters = recovered_injection_parameters, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'PAIRSvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 2D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - PAIRSvSINGLES (2D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 4D Fitting Factor for SINGLES')
ht = gen.wf_td(injection_parameters = waveform_metadata_a['H1']['parameters'], **kwargs)
wf_model = "4D"  # ["4D", "EC_5D", "ML_6D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None, 
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2 = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2
recovered_injection_parameters['spin_1z'] = recovered_Sz1
recovered_injection_parameters['spin_2z'] = recovered_Sz2

ht_recovered = gen.wf_td(injection_parameters = recovered_injection_parameters, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'SINGLESvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - SINGLESvSINGLES (4D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 4D Fitting Factor for PAIRS')
ht = gen.pairs_td(injection_parameters_a = waveform_metadata_a['H1']['parameters'], injection_parameters_b = waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "4D"  # ["4D", "EC_5D", "ML_6D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None, 
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2 = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2
recovered_injection_parameters['spin_1z'] = recovered_Sz1
recovered_injection_parameters['spin_2z'] = recovered_Sz2

ht_recovered = gen.wf_td(injection_parameters = recovered_injection_parameters, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'PAIRSvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - PAIRSvSINGLES (4D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 3D Eccentric Fitting Factor for SINGLES')
ht = gen.wf_ecc_td(injection_parameters = waveform_metadata_a['H1']['parameters'], e = 0, **kwargs)
wf_model = "EC_3D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, e): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_e = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

ht_recovered = gen.wf_ecc_td(injection_parameters = recovered_injection_parameters, e=recovered_e, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'SINGLESvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 3D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_e))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - SINGLESvECC (3D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 3D Eccentric Fitting Factor for ECC')
ht = gen.wf_ecc_td(injection_parameters = waveform_metadata_a['H1']['parameters'], e = 0.3, **kwargs)
wf_model = "EC_3D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, e): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_e = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

ht_recovered = gen.wf_ecc_td(injection_parameters = recovered_injection_parameters, e=recovered_e, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'ECCvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 3D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_e))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - ECCvECC (3D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 3D Eccentric Fitting Factor for PAIRS')
ht = gen.pairs_ecc_td(injection_parameters_a = waveform_metadata_a['H1']['parameters'], injection_parameters_b = waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "EC_3D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, e): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_e = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

ht_recovered = gen.wf_ecc_td(injection_parameters = recovered_injection_parameters, e=recovered_e, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'PAIRSvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 3D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_e))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - PAIRSvECC (3D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 5D Eccentric Fitting Factor for PAIRS')
ht = gen.pairs_ecc_td(injection_parameters_a = waveform_metadata_a['H1']['parameters'], injection_parameters_b = waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "EC_5D"  # ["4D", "EC_5D", "ML_6D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None, 
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2, e): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_e = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2
recovered_injection_parameters['spin_1z'] = recovered_Sz1
recovered_injection_parameters['spin_2z'] = recovered_Sz2

ht_recovered = gen.wf_ecc_td(injection_parameters = recovered_injection_parameters, e=recovered_e, **kwargs)
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'PAIRSvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 5D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$, $e = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_e))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - PAIRSvECC (5D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 4D Microlensed Fitting Factor for SINGLES')
ht = gen.wf_ml_fd(injection_parameters = waveform_metadata_a['H1']['parameters'], Ml_z = 0, y = 0, **kwargs)
wf_model = "ML_4D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Mlz, y): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

hf_recovered = gen.wf_ml_fd(injection_parameters = recovered_injection_parameters, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}
ht = {'H1': ht['H1'].to_timeseries(), 'L1': ht['L1'].to_timeseries(), 'V1': ht['V1'].to_timeseries()}
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_injection_parameters['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'SINGLESvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $M_{{lz}} = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Mlz, recovered_y))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - SINGLESvML (4D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 4D Microlensed Fitting Factor for ML')
ht = gen.wf_ml_fd(injection_parameters = waveform_metadata_a['H1']['parameters'], Ml_z = 1e3, y = 0.1, **kwargs)
wf_model = "ML_4D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Mlz, y): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

hf_recovered = gen.wf_ml_fd(injection_parameters = recovered_injection_parameters, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}
ht = {'H1': ht['H1'].to_timeseries(), 'L1': ht['L1'].to_timeseries(), 'V1': ht['V1'].to_timeseries()}
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_injection_parameters['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'MLvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $M_{{lz}} = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Mlz, recovered_y))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - MLvML (4D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 4D Microlensed Fitting Factor for PAIRS')
ht = gen.pairs_ml_fd(injection_parameters_a = waveform_metadata_a['H1']['parameters'], injection_parameters_b = waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "ML_4D"  # ["2D", "EC_3D", "ML_4D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None,
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Mlz, y): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2

hf_recovered = gen.wf_ml_fd(injection_parameters = recovered_injection_parameters, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}
ht = {'H1': ht['H1'].to_timeseries(), 'L1': ht['L1'].to_timeseries(), 'V1': ht['V1'].to_timeseries()}
for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_injection_parameters['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'PAIRSvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $M_{{lz}} = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Mlz, recovered_y))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - PAIRSvML (4D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################

print('\nComputing 6D Microlensed Fitting Factor for PAIRS')
ht = gen.pairs_ml_fd(injection_parameters_a = waveform_metadata_a['H1']['parameters'], injection_parameters_b = waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "ML_6D"  # ["4D", "EC_5D", "ML_6D"]
FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM" , f_lower=f_lower, f_high=f_high, psd=None, 
                                n_iters=['default'], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]) )
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2, Mlz, y): ', FF_res[1])

##############################################################################################################################################################################

recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_injection_parameters = waveform_metadata_a['H1']['parameters'].copy()

recovered_injection_parameters['mass_1'] = recovered_mass_1
recovered_injection_parameters['mass_2'] = recovered_mass_2
recovered_injection_parameters['spin_1z'] = recovered_Sz1
recovered_injection_parameters['spin_2z'] = recovered_Sz2

hf_recovered = gen.wf_ml_fd(injection_parameters = recovered_injection_parameters, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}
ht = {'H1': ht['H1'].to_timeseries(), 'L1': ht['L1'].to_timeseries(), 'V1': ht['V1'].to_timeseries()}

for key in ht.keys():
    ht[key].start_time += ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))] - ht[key].sample_times[np.argmax(np.abs(ht[key]))]
    ht_recovered[key].start_time += ht[key].sample_times[np.argmax(np.abs(ht[key]))] - ht_recovered[key].sample_times[np.argmax(np.abs(ht_recovered[key]))]

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['L1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_injection_parameters['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_injection_parameters['geocent_time'] - 0.5, recovered_injection_parameters['geocent_time'] + 0.3)
plt.title(r'PAIRSvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 6D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$, $\mathcal{{M}}_{{\ell}}^z = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_Mlz, recovered_y))
plt.savefig('/home/nishkal.rao/git_overlap/src/output/match_final/Recovered Waveform - PAIRSvML (6D).png', bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
##############################################################################################################################################################################
