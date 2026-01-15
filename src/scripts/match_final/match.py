import importlib
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('/home/nishkal.rao/git_overlap/src/scripts/match_final/')

import FF_computation as ff
import waveforms as wf
from FF_computation import mchirp_q_to_m1m2

plt.style.use('default')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

output_dir = '/home/nishkal.rao/git_overlap/src/output/match_final/'
os.makedirs(output_dir, exist_ok=True)

gen = wf.PairsWaveformGeneration()

# Setup Parameters
f_lower = 20.0
f_ref = 50.0
sampling_frequency = 4096.0
f_high = sampling_frequency / 2.0
delta_t = 1.0 / sampling_frequency

# Standard kwargs for waveform generation
kwargs = dict(sampling_frequency=sampling_frequency, f_lower=f_lower, f_ref=f_ref, f_high=f_high, delta_t=delta_t)

# Load Injection Data
with open('/home/nishkal.rao/git_overlap/src/output/injections/GW Waveform A Meta Data.pkl', 'rb') as f:
    waveform_metadata_a = pickle.load(f)

with open('/home/nishkal.rao/git_overlap/src/output/injections/GW Waveform B Meta Data.pkl', 'rb') as f:
    waveform_metadata_b = pickle.load(f)
    
ff_config = dict(
    f_low=f_lower, 
    f_high=f_high, 
    psd=None, 
    n_iters=100,
)

##############################################################################################################################################################################
# 1. SINGLES v SINGLES (2D)
##############################################################################################################################################################################

print('\nComputing 2D Fitting Factor for SINGLES')
ht = gen.wf_td(injection_parameters=waveform_metadata_a['H1']['parameters'], **kwargs)
wf_model = "2D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

ht_recovered = gen.wf_td(injection_parameters=recovered_prms, **kwargs)

# Align waveforms peak-to-peak for visualization
shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'SINGLESvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 2D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - SINGLESvSINGLES (2D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 2. PAIRS v SINGLES (2D)
##############################################################################################################################################################################

print('\nComputing 2D Fitting Factor for PAIRS')
ht = gen.pairs_td(injection_parameters_a=waveform_metadata_a['H1']['parameters'], 
                  injection_parameters_b=waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "2D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

ht_recovered = gen.wf_td(injection_parameters=recovered_prms, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform (Pairs)')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform (Single)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'PAIRSvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 2D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - PAIRSvSINGLES (2D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 3. SINGLES v SINGLES (4D)
##############################################################################################################################################################################

print('\nComputing 4D Fitting Factor for SINGLES')
ht = gen.wf_td(injection_parameters=waveform_metadata_a['H1']['parameters'], **kwargs)
wf_model = "4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2 = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2, spin_1z=recovered_Sz1, spin_2z=recovered_Sz2)

ht_recovered = gen.wf_td(injection_parameters=recovered_prms, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'SINGLESvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - SINGLESvSINGLES (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 4. PAIRS v SINGLES (4D)
##############################################################################################################################################################################

print('\nComputing 4D Fitting Factor for PAIRS')
ht = gen.pairs_td(injection_parameters_a=waveform_metadata_a['H1']['parameters'], 
                  injection_parameters_b=waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2 = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2, spin_1z=recovered_Sz1, spin_2z=recovered_Sz2)

ht_recovered = gen.wf_td(injection_parameters=recovered_prms, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'PAIRSvSINGLES: Recovered Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - PAIRSvSINGLES (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 5. SINGLES v ECC (4D)
##############################################################################################################################################################################

print('\nComputing 4D Eccentric Fitting Factor for SINGLES')
ht = gen.wf_ecc_td(injection_parameters=waveform_metadata_a['H1']['parameters'], e=0, anomaly=0, **kwargs)
wf_model = "EC_4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, e, anomaly): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_e, recovered_anomaly = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

ht_recovered = gen.wf_ecc_td(injection_parameters=recovered_prms, e=recovered_e, anomaly=recovered_anomaly, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'SINGLESvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$, $\text{{anomaly}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_e, recovered_anomaly))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - SINGLESvECC (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 6. ECC v ECC (4D)
##############################################################################################################################################################################

print('\nComputing 4D Eccentric Fitting Factor for ECC')
ht = gen.wf_ecc_td(injection_parameters=waveform_metadata_a['H1']['parameters'], e=0.3, anomaly=np.pi/4, **kwargs)
wf_model = "EC_4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, e, anomaly): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_e, recovered_anomaly = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

# Assuming anomaly matches injection for comparison, though search doesn't optimize it explicitly in 3D
ht_recovered = gen.wf_ecc_td(injection_parameters=recovered_prms, e=recovered_e, anomaly=recovered_anomaly, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'ECCvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$, $\text{{anomaly}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_e, recovered_anomaly))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - ECCvECC (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 7. PAIRS v ECC (4D)
##############################################################################################################################################################################

print('\nComputing 4D Eccentric Fitting Factor for PAIRS')
ht = gen.pairs_ecc_td(injection_parameters_a=waveform_metadata_a['H1']['parameters'], 
                      injection_parameters_b=waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "EC_4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, e, anomaly): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_e, recovered_anomaly = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

ht_recovered = gen.wf_ecc_td(injection_parameters=recovered_prms, e=recovered_e, anomaly=recovered_anomaly, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'PAIRSvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $e = {:.3f}$, $\text{{anomaly}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_e, recovered_anomaly))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - PAIRSvECC (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 8. PAIRS v ECC (6D)
##############################################################################################################################################################################

print('\nComputing 6D Eccentric Fitting Factor for PAIRS')
ht = gen.pairs_ecc_td(injection_parameters_a=waveform_metadata_a['H1']['parameters'], 
                      injection_parameters_b=waveform_metadata_b['H1']['parameters'], **kwargs)
wf_model = "EC_6D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2, e, anomaly): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_e, recovered_anomaly = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2, spin_1z=recovered_Sz1, spin_2z=recovered_Sz2)

ht_recovered = gen.wf_ecc_td(injection_parameters=recovered_prms, e=recovered_e, anomaly=recovered_anomaly, **kwargs)

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'PAIRSvECC: Recovered Eccentric Waveform with Match: ${:.3f}$ and 6D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$, $e = {:.3f}$, $\text{{anomaly}} = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_e, recovered_anomaly))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - PAIRSvECC (6D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 9. SINGLES v ML (4D)
##############################################################################################################################################################################

print('\nComputing 4D Microlensed Fitting Factor for SINGLES')
# Note: gen.wf_ml_fd returns FrequencySeries dict.
hf = gen.wf_ml_fd(injection_parameters=waveform_metadata_a['H1']['parameters'], Ml_z=0, y=0, **kwargs)
ht = {'H1': hf['H1'].to_timeseries(), 'L1': hf['L1'].to_timeseries(), 'V1': hf['V1'].to_timeseries()}
wf_model = "ML_4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Mlz, y): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

hf_recovered = gen.wf_ml_fd(injection_parameters=recovered_prms, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
# Update geocent time based on peak for plotting focus
recovered_prms['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'SINGLESvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $M_{{lz}} = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Mlz, recovered_y))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - SINGLESvML (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 10. ML v ML (4D)
##############################################################################################################################################################################

print('\nComputing 4D Microlensed Fitting Factor for ML')
hf = gen.wf_ml_fd(injection_parameters=waveform_metadata_a['H1']['parameters'], Ml_z=1e3, y=0.1, **kwargs)
ht = {'H1': hf['H1'].to_timeseries(), 'L1': hf['L1'].to_timeseries(), 'V1': hf['V1'].to_timeseries()}
wf_model = "ML_4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Mlz, y): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

hf_recovered = gen.wf_ml_fd(injection_parameters=recovered_prms, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_prms['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'MLvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $M_{{lz}} = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Mlz, recovered_y))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - MLvML (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 11. PAIRS v ML (4D)
##############################################################################################################################################################################

print('\nComputing 4D Microlensed Fitting Factor for PAIRS')
hf = gen.pairs_ml_fd(injection_parameters_a=waveform_metadata_a['H1']['parameters'], 
                     injection_parameters_b=waveform_metadata_b['H1']['parameters'], **kwargs)
ht = {'H1': hf['H1'].to_timeseries(), 'L1': hf['L1'].to_timeseries(), 'V1': hf['V1'].to_timeseries()}
wf_model = "ML_4D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Mlz, y): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2)

hf_recovered = gen.wf_ml_fd(injection_parameters=recovered_prms, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_prms['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'PAIRSvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 4D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $M_{{lz}} = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Mlz, recovered_y))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - PAIRSvML (4D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################
# 12. PAIRS v ML (6D)
##############################################################################################################################################################################

print('\nComputing 6D Microlensed Fitting Factor for PAIRS')
hf = gen.pairs_ml_fd(injection_parameters_a=waveform_metadata_a['H1']['parameters'], 
                     injection_parameters_b=waveform_metadata_b['H1']['parameters'], **kwargs)
ht = {'H1': hf['H1'].to_timeseries(), 'L1': hf['L1'].to_timeseries(), 'V1': hf['V1'].to_timeseries()}
wf_model = "ML_6D"

FF_res = ff.compute_fitting_factor(ht['H1'], wf_model=wf_model, apx="IMRPhenomXPHM", **ff_config)
print('Best recovered match (FF): {:.3f}'.format(FF_res[0]))
print('Parameters corresponding to the best matched WF (M_chirp, q, Sz1, Sz2, Mlz, y): ', FF_res[1])

# Recover and Plot
recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_Mlz, recovered_y = FF_res[1]
recovered_mass_1, recovered_mass_2 = mchirp_q_to_m1m2(recovered_mchirp, recovered_q)
recovered_prms = waveform_metadata_a['H1']['parameters'].copy()
recovered_prms.update(mass_1=recovered_mass_1, mass_2=recovered_mass_2, spin_1z=recovered_Sz1, spin_2z=recovered_Sz2)

hf_recovered = gen.wf_ml_fd(injection_parameters=recovered_prms, Ml_z=recovered_Mlz, y=recovered_y, **kwargs)
ht_recovered = {'H1': hf_recovered['H1'].to_timeseries(), 'L1': hf_recovered['L1'].to_timeseries(), 'V1': hf_recovered['V1'].to_timeseries()}

shift = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))] - ht['H1'].sample_times[np.argmax(np.abs(ht['H1']))]
for key in ht.keys():
    ht[key].start_time += shift

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(ht['H1'].sample_times, ht['H1'], label='Injected Waveform')
ax.plot(ht_recovered['H1'].sample_times, ht_recovered['H1'], label='Recovered Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend()
recovered_prms['geocent_time'] = ht_recovered['H1'].sample_times[np.argmax(np.abs(ht_recovered['H1']))]
ax.set_xlim(recovered_prms['geocent_time'] - 0.5, recovered_prms['geocent_time'] + 0.3)
plt.title(r'PAIRSvML: Recovered Microlensed Waveform with Match: ${:.3f}$ and 6D Parameters $\mathcal{{M}} = {:.3f}$, $q = {:.3f}$, $S_{{z1}} = {:.3f}$, $S_{{z2}} = {:.3f}$, $\mathcal{{M}}_{{\ell}}^z = {:.3f}$, $y = {:.3f}$'.format(FF_res[0], recovered_mchirp, recovered_q, recovered_Sz1, recovered_Sz2, recovered_Mlz, recovered_y))
plt.savefig(os.path.join(output_dir, 'Recovered Waveform - PAIRSvML (6D).png'), bbox_inches='tight')
plt.close()

##############################################################################################################################################################################