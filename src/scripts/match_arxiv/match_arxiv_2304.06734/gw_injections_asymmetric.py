# https://arxiv.org/pdf/2304.06734.pdf
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
nmap_a = np.argmax(injection_a['log_likelihood']+injection_a['log_prior'])   # Maximum A Posteriori values

injection_parameters_a = {
    "mass_1": 30, # Primary redshifted mass of the heavier binary component in solar masses
    "mass_2": 20, # Secondary redshifted mass of the lighter binary component in solar masses
    "luminosity_distance": 500, # Distance in Megaparsecs
    "a_1": injection_a['a_1'][nmap_a], # Dimensionless spin of the heavier black hole
    "a_2": injection_a['a_2'][nmap_a], # Dimensionless spin of the lighter black hole    "ra": injection_a['ra'][nmap_a], # Right ascension of the source, in degrees
    "ra": injection_a['ra'][nmap_a], # Right Asencion of the source, in radians
    "dec": injection_a['dec'][nmap_a], # Declination of the source, in radians
    "tilt_1": injection_a['tilt_1'][nmap_a], # Tilt angle of heavier black hole, in radians
    "tilt_2": injection_a['tilt_2'][nmap_a], # Tilt angle of lighter black hole, in radians
    "theta_jn": injection_a['theta_jn'][nmap_a], # Angle between the total angular momentum and line of sight
    "phase": injection_a['phase'][nmap_a], # Orbital phase at coalescence
    "psi": injection_a['psi'][nmap_a], # Polarization angle
    "incl": injection_a['iota'][nmap_a], # Inclination of the orbital angular momentum with respect to the line of sight
    "geocent_time": injection_a['geocent_time'][nmap_a] # GPS time of the event
}

N_sampl = 6
injection_parameters_b = []
delta_tc = [-0.0131, -0.0043, -0.0014, 0.0010, 0.0031, 0.0347]
for i in range(N_sampl):
    injection_parameters_b.append({
        "mass_1": 27, # Primary redshifted mass of the heavier binary component in solar masses
        "mass_2": 18, # Secondary redshifted mass of the lighter binary component in solar masses
        "luminosity_distance": 3000, # Distance in Megaparsecs        "a_1": injection_a['a_1'][nmap_a], # Dimensionless spin of the heavier black hole
        "a_1": injection_a['a_1'][nmap_a], # Dimensionless spin of the heavier black hole        "ra": injection_a['ra'][nmap_a], # Right ascension of the source, in degrees
        "a_2": injection_a['a_2'][nmap_a], # Dimensionless spin of the lighter black hole        "ra": injection_a['ra'][nmap_a], # Right ascension of the source, in degrees
        "ra": injection_a['ra'][nmap_a], # Right Asencion of the source, in radians
        "dec": injection_a['dec'][nmap_a], # Declination of the source, in radians
        "tilt_1": injection_a['tilt_1'][nmap_a], # Tilt angle of heavier black hole, in radians
        "tilt_2": injection_a['tilt_2'][nmap_a], # Tilt angle of lighter black hole, in radians
        "theta_jn": injection_a['theta_jn'][nmap_a], # Angle between the total angular momentum and line of sight
        "phase": injection_a['phase'][nmap_a], # Orbital phase at coalescence
        "psi": injection_a['psi'][nmap_a], # Polarization angle
        "incl": injection_a['iota'][nmap_a], # Inclination of the orbital angular momentum with respect to the line of sight
        "geocent_time": injection_a['geocent_time'][nmap_a]+delta_tc[i] # GPS time of the event
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
with open('git_overlap/src/output/match_arxiv/match_asymmetric/injections/GW Waveform A Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_a.meta_data, file) 

for i in range(N_sampl):
    ifos_b.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b[i])    # SINGLES B
    with open('git_overlap/src/output/match_arxiv/match_asymmetric/injections/GW Waveform B Meta Data %s.pkl'%(i+1), 'wb') as file:
        pickle.dump(ifos_b.meta_data, file)