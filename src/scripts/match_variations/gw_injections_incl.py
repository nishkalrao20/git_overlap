# Generate BBH Injections (SINGLES/ PAIRS) resembling merger events and inject into detectors (H1, L1, V1) 

import bilby
import pickle
import numpy as np
from pesummary.io import read
import matplotlib.pyplot as plt

# Preamble

delta_f = 1
duration = 100
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

N_sampl = 25
injection_parameters_b = []
incl = np.random.uniform(0,np.pi, N_sampl)
geocent_time_b = injection_a['geocent_time'][nmap_a]+np.random.uniform(0,0.5)
for i in range(N_sampl):
    injection_parameters_b.append({
        "mass_1_source": injection_b['mass_1_source'][nmap_b], # Source-frame mass of the heavier binary component in solar masses
        "mass_ratio": injection_b['mass_ratio'][nmap_b], # Mass ratio of the binary
        "mass_1": injection_b['mass_1'][nmap_b], # Primary redshifted mass of the heavier binary component in solar masses
        "mass_2": injection_b['mass_2'][nmap_b], # Secondary redshifted mass of the lighter binary component in solar masses
        "luminosity_distance": injection_b['luminosity_distance'][nmap_b]*(i+1), # Distance in Megaparsecs
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
        "phase": injection_b['iota'][nmap_b], # Orbital phase at coalescence
        "incl": incl[i], # Inclination of the orbital angular momentum with respect to the line of sight
        "geocent_time": geocent_time_b # GPS time of the event
    })

start_time = injection_parameters_a['geocent_time']-duration+2

# Bilby's WaveformGenerator object to generate BBH waveforms

waveform_generator_a = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, start_time = start_time,
                                                  frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments = {'waveform_approximant': 'IMRPhenomXPHM', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

waveform_generator_b = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency = sampling_frequency, start_time = start_time,
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
with open('git_overlap/src/output/match_variations/match_incl/injections/GW Waveform A Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_a.meta_data, file) 

for i in range(N_sampl):
    ifos_b.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b[i])    # SINGLES B
    with open('git_overlap/src/output/match_variations/match_incl/injections/GW Waveform B Meta Data %s.pkl'%(i+1), 'wb') as file:
        pickle.dump(ifos_b.meta_data, file)