# Parameter Estimation runs of the SINGLES waveforms

import bilby
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generating the priors dictionary

priors = bilby.prior.PriorDict(
    dict(
        chirp_mass = bilby.core.prior.Uniform(name='chirp_mass', minimum=12.299703, maximum=45, unit='$M_{\odot}$'),
        mass_ratio = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.125, maximum=1),
        mass_1 = bilby.core.prior.Constraint(name='mass_1', minimum=1.001398, maximum=1000),
        mass_2 = bilby.core.prior.Constraint(name='mass_2', minimum=1.001398, maximum=1000),
        a_1 = bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.88),
        a_2 = bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.88),
        tilt_1 = bilby.core.prior.Sine(name='tilt_1'),
        tilt_2 = bilby.core.prior.Sine(name='tilt_2'),
        phi_12 = bilby.core.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        phi_jl = bilby.core.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        luminosity_distance = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=5e3, unit='Mpc'),
        dec = bilby.core.prior.Cosine(name='dec'),
        ra = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        theta_jn = bilby.core.prior.Sine(name='theta_jn'),
        psi = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
        phase = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
    )
)

# Setting some constants

duration = 100
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

def pe(injection_parameters):
    """Initialises the likelihood by passing in the interferometer data and updates the priors based on the injection parameters"""

    waveform_generator = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency, start_time=injection_parameters['geocent_time']-duration+2,
                                                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                    waveform_arguments={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])   # Initializing the Detectors

    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = minimum_frequency, maximum_frequency
    ifos.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=injection_parameters['geocent_time']-duration+2)

    # Generating the likelihood and running the sampler

    priors['geocent_time'] = injection_parameters['geocent_time']
    likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
                                                    time_marginalization=False, phase_marginalization=False, distance_marginalization=False)

    return likelihood, priors

# Importing the waveform metadata

waveform_metadata_a, waveform_metadata_b = pickle.load(open('git_overlap/src/output/injections/GW Waveform A Meta Data.pkl', 'rb')), pickle.load(open('git_overlap/src/output/injections/GW Waveform B Meta Data.pkl', 'rb'))   # Importing Waveform Meta Data

# Running the sampler for the SINGLES waveformms

likelihood_a, priors_a = pe(waveform_metadata_a['H1']['parameters'])
result_a = bilby.run_sampler(
        likelihood_a, priors_a, sampler = 'dynesty', outdir = 'git_overlap/src/output/pe/bilby/singles_a', label = 'SIGNLES_A',
        nlive=1000, check_point_delta_t=600, check_point_plot=True, npool=1,
        conversion_function = bilby.gw.conversion.generate_all_bbh_parameters,
    )
result_a.plot_corner()

likelihood_b, priors_b = pe(waveform_metadata_b['H1']['parameters'])
result_b = bilby.run_sampler(
        likelihood_b, priors_b, sampler = 'dynesty', outdir = 'git_overlap/src/output/pe/bilby/singles_b', label = 'SIGNLES_B',
        nlive=1000, check_point_delta_t=600, check_point_plot=True, npool=1,
        conversion_function = bilby.gw.conversion.generate_all_bbh_parameters
    )
result_b.plot_corner() 