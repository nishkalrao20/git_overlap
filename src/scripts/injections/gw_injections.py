# Generate BBH Injections (SINGLES/ PAIRS) resembling merger events and inject into detectors (H1, L1, V1) 

import bilby
import pycbc
import pickle
import numpy as np
import pycbc.types
from pycbc import frame
from pesummary.io import read
import matplotlib.pyplot as plt
from pycbc.waveform.utils import taper_timeseries

# Preamble

delta_f = 1
duration = 4
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

# Reading the posterior samples from GWTC release, and indexing the MAP values (https://zenodo.org/record/6513631)

injection_a = read('git_overlap/src/data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']   # Loading the GW150914 Posterior distributions
injection_b = read('git_overlap/src/data/IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_cosmo.h5').samples_dict['C01:IMRPhenomXPHM']   # Loading the GW170814 Posterior distributions

nmap_a = np.argmax(injection_a['log_likelihood']+injection_a['log_prior'])   # Maximum A Posteriori values
nmap_b = np.argmax(injection_b['log_likelihood']+injection_b['log_prior'])   # Maximum A Posteriori values

injection_parameters_a = {
    "mass_1_source": injection_a['mass_1_source'][nmap_a], # Source-frame mass of the heavier binary component in solar masses
    "mass_ratio": injection_a['mass_ratio'][nmap_a], # Mass ratio of the binary
    "mass_1": injection_a['mass_1'][nmap_a], # Primary redshifted mass of the heavier binary component in solar masses
    "mass_2": injection_a['mass_2'][nmap_a], # Secondary redshifted mass of the lighter binary component in solar masses
    "luminosity_distance": injection_a['luminosity_distance'][nmap_a], # Distance in Megaparsecs
    "redshift": injection_a['redshift'][nmap_a], # Cosmological redshift of the source
    "chi_eff": injection_a["chi_eff"][nmap_a], # Effective spin parameter 
    "a_1": injection_a['a_1'][nmap_a], # Dimensionless spin of the heavier black hole
    "a_2": injection_a['a_2'][nmap_a], # Dimensionless spin of the lighter black hole
    "tilt_1": injection_a['tilt_1'][nmap_a], # Tilt angle of heavier black hole, in radians
    "tilt_2": injection_a['tilt_2'][nmap_a], # Tilt angle of lighter black hole, in radians
    "theta_jn": injection_a['theta_jn'][nmap_a], # Angle between the total angular momentum and line of sight
    "phi_12": injection_a['phi_12'][nmap_a], # Azimuthal angle between the spins of the black holes
    "phi_jl": injection_a['phi_jl'][nmap_a], # Azimuthal angle between the total angular momentum of the binary and the direction to the observer
    "ra": injection_a['ra'][nmap_a], # Right ascension of the source, in degrees
    "dec": injection_a['dec'][nmap_a], # Declination of the source, in radians
    "psi": injection_a['psi'][nmap_a], # Polarization angle
    "phase": injection_a['phase'][nmap_a], # Orbital phase at coalescence
    "incl": injection_a['iota'][nmap_a], # Inclination of the orbital angular momentum with respect to the line of sight
    "geocent_time": injection_a['geocent_time'][nmap_a] # GPS time of the event
}

injection_parameters_b = {
    "mass_1_source": injection_b['mass_1_source'][nmap_b], # Source-frame mass of the heavier binary component in solar masses
    "mass_ratio": injection_b['mass_ratio'][nmap_b], # Mass ratio of the binary
    "mass_1": injection_b['mass_1'][nmap_b], # Primary redshifted mass of the heavier binary component in solar masses
    "mass_2": injection_b['mass_2'][nmap_b], # Secondary redshifted mass of the lighter binary component in solar masses
    "luminosity_distance": injection_b['luminosity_distance'][nmap_b], # Distance in Megaparsecs
    "redshift": injection_b['redshift'][nmap_b], # Cosmological redshift of the source
    "chi_eff": injection_b["chi_eff"][nmap_b], # Effective spin parameter 
    "a_1": injection_b['a_1'][nmap_b], # Dimensionless spin of the heavier black hole
    "a_2": injection_b['a_2'][nmap_b], # Dimensionless spin of the lighter black hole
    "tilt_1": injection_b['tilt_1'][nmap_b], # Tilt angle of heavier black hole, in radians
    "tilt_2": injection_b['tilt_2'][nmap_b], # Tilt angle of lighter black hole, in radians
    "theta_jn": injection_b['theta_jn'][nmap_b], # Angle between the total angular momentum and line of sight
    "phi_12": injection_b['phi_12'][nmap_b], # Azimuthal angle between the spins of the black holes
    "phi_jl": injection_b['phi_jl'][nmap_b], # Azimuthal angle between the total angular momentum of the binary and the direction to the observer
    "ra": injection_b['ra'][nmap_b], # Right ascension of the source, in degrees
    "dec": injection_b['dec'][nmap_b], # Declination of the source, in radians
    "psi": injection_b['psi'][nmap_b], # Polarization angle
    "phase": injection_b['phase'][nmap_b], # Orbital phase at coalescence
    "incl": injection_b['iota'][nmap_b], # Inclination of the orbital angular momentum with respect to the line of sight
    "geocent_time": injection_a['geocent_time'][nmap_a]+np.random.uniform(0,0.5) # GPS time of the event
}

start_time = injection_parameters_a['geocent_time']-duration+2

# Bilby's WaveformGenerator object to generate BBH waveforms

waveform_generator_a = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, start_time = start_time,
                                                  frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments = {'waveform_approximant': 'IMRPhenomXPHM', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

waveform_generator_b = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency = sampling_frequency, start_time = start_time,
                                                  frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments = {'waveform_approximant': 'IMRPhenomXPHM', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

waveform_generator = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency = sampling_frequency, start_time = start_time,
                                                  frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments = {'waveform_approximant': 'IMRPhenomXPHM', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

# Initializing the detectors as bilby interferometer objects with zero noise, and with GPS time around the geocenter time of the GW signal

ifos, ifos_a, ifos_b = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])   # Initialize Detectors

for det in [ifos, ifos_a, ifos_b]:
    for ifo in det:
        ifo.minimum_frequency, ifo.maximum_frequency  = minimum_frequency, sampling_frequency/2
    det.set_strain_data_from_zero_noise(sampling_frequency = sampling_frequency, duration = duration, start_time = start_time)

# Injecting the SINGLES GW signal into H1, L1, and V1 using bilby, and saving the parameters

ifos_a.inject_signal(waveform_generator = waveform_generator_a, parameters = injection_parameters_a)    # SINGLES A
with open('git_overlap/src/output/injections/GW Waveform A Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_a.meta_data, file) 

ifos_b.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b)    # SINGLES B
with open('git_overlap/src/output/injections/GW Waveform B Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_b.meta_data, file)

# Injecting the PAIRS GW signal into H1, L1, and V1 using bilby

ifos.inject_signal(waveform_generator = waveform_generator_a, parameters = injection_parameters_a)    # PAIRS (A)
ifos.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b)    # PAIRS (B)

# Extrapolating the strain data in the time domain

H1_strain_a, L1_strain_a, V1_strain_a = ifos_a[0].time_domain_strain, ifos_a[1].time_domain_strain, ifos_a[2].time_domain_strain    # SINGLES A
H1_strain_b, L1_strain_b, V1_strain_b = ifos_b[0].time_domain_strain, ifos_b[1].time_domain_strain, ifos_b[2].time_domain_strain    # SINGLES B
H1_strain, L1_strain, V1_strain = ifos[0].time_domain_strain, ifos[1].time_domain_strain, ifos[2].time_domain_strain    # PAIRS

# Generating PyCBC TimeSeries from the strain array

ht_a_H1, ht_a_L1, ht_a_V1 = pycbc.types.TimeSeries(H1_strain_a, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain_a, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain_a, delta_t = 1/sampling_frequency)    # SINGLES A
ht_b_H1, ht_b_L1, ht_b_V1 = pycbc.types.TimeSeries(H1_strain_b, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain_b, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain_b, delta_t = 1/sampling_frequency)    # SINGLES B
ht_H1, ht_L1, ht_V1 = pycbc.types.TimeSeries(H1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain, delta_t = 1/sampling_frequency)    # PAIRS

# Setting the start times to the geocenter time and creating the dictionary of waveforms 

ht_a_H1.start_time, ht_a_L1.start_time, ht_a_V1.start_time = start_time, start_time, start_time    # SINGLES A
ht_b_H1.start_time, ht_b_L1.start_time, ht_b_V1.start_time = start_time, start_time, start_time    # SINGLES B
ht_H1.start_time, ht_L1.start_time, ht_V1.start_time = start_time, start_time, start_time    # PAIRS

ht_a, ht_b, ht = {'H1': ht_a_H1, 'L1': ht_a_L1, 'V1': ht_a_V1}, {'H1': ht_b_H1, 'L1': ht_b_L1, 'V1': ht_b_V1}, {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

# Matplotlib rcParams

plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

# Plotting the waveforms

fig, ax = plt.subplots(1, 3, figsize = (18, 5), sharex = True)
fig.suptitle('\\textbf{WAVEFORM INJECTIONS: SINGLES A (GW150914) and SINGLES B (GW170814)}')

ax[0].plot(ht_a['H1'].sample_times, ht_a['H1'], 'g-', linewidth=1, label='$\\rm{SINGLES_A}$')
ax[0].plot(ht_b['H1'].sample_times, ht_b['H1'], 'm-', linewidth=1, label='$\\rm{SINGLES_B}$')
ax[0].plot(ht['H1'].sample_times, ht['H1'], 'b--', label='$\\rm{PAIRS}$')
ax[0].set_xlabel('Time $[s]$')
ax[0].set_ylabel('Strain $h$')
ax[0].set_xlim(start_time+duration-2.4, start_time+duration-1.7)
ax[0].legend() 
ax[0].grid(True)
ax[0].set_title('H1')

ax[1].plot(ht_a['L1'].sample_times, ht_a['L1'], 'g-', linewidth=1, label='$\\rm{SINGLES_A}$')
ax[1].plot(ht_b['L1'].sample_times, ht_b['L1'], 'm-', linewidth=1, label='$\\rm{SINGLES_B}$')
ax[1].plot(ht['L1'].sample_times, ht['L1'], 'b--', label='$\\rm{PAIRS}$')
ax[1].set_xlabel('Time $[s]$')
ax[1].set_ylabel('Strain $h$')
ax[1].set_xlim(start_time+duration-2.4, start_time+duration-1.7)
ax[1].legend() 
ax[1].grid(True)
ax[1].set_title('L1')

ax[2].plot(ht_a['V1'].sample_times, ht_a['V1'], 'g-', linewidth=1, label='$\\rm{SINGLES_A}$')
ax[2].plot(ht_b['V1'].sample_times, ht_b['V1'], 'm-', linewidth=1, label='$\\rm{SINGLES_B}$')
ax[2].plot(ht['H1'].sample_times, ht['V1'], 'b--', label='$\\rm{PAIRS}$')
ax[2].set_xlabel('Time $[s]$')
ax[2].set_ylabel('Strain $h$')
ax[2].set_xlim(start_time+duration-2.4, start_time+duration-1.7)
ax[2].legend() 
ax[2].grid(True)
ax[2].set_title('V1')

fig.set_tight_layout(True)
plt.savefig('git_overlap/src/output/injections/GW INJECTIONS.png')
plt.close()
